import os
import json
import logging
import traceback
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage

from awb.llm_agent import LLMAgentParser
from awb.local_agent import LocalAgentParser
from awb.vision_agent import VisionAgentParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # This should not be required in other environment
# path = poppler_path + os.pathsep + tessaract_path # Example path for Windows
# os.environ["PATH"] += os.pathsep + path


class OCRParser:
    def __init__(self):
        self.tasks = {
            "local": "Extract the data in the file using the Local OCR Tools. Return the data in structured JSON format for Airway bills.",
            "llm": "Extract the data in the file using the LLM Based OCR Tools.",
            "vision": "Extract the data in the file using the AI Computer Vision Based OCR Tools.",
        }
    
    async def run_awb_parser_agent(self, parser, task_description):
        """Runs the given AWB parsing agent with a specified task description."""
        logging.info(f"Parser Started : {parser.agent.name}")
        
        try:
            response = await parser.agent.on_messages(
                [TextMessage(content=task_description, source="user")],
                cancellation_token=CancellationToken(),
            )
            
            if response.chat_message and response.chat_message.content:
                try:
                    structured_data = json.loads(response.chat_message.content)
                    return structured_data
                except json.JSONDecodeError:
                    logging.error(f"Response from {parser.agent.name} is not valid JSON.")
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Parser failed - {parser.agent.name} : {e}")
        
        return {}
    
    def count_filled_fields(self, json_data, reference_data=None):
        """Counts non-empty, non-null, and non-'Unknown' fields in the JSON response."""
        if not json_data:
            return 0

        def is_valid(value):
            return value is not None and value != "" and value != "Unknown"

        def compare_with_reference(value, ref_value):
            if reference_data:
                return is_valid(value) and value == ref_value
            return is_valid(value)

        def recursive_count(data, ref=None):
            count = 0
            for key, value in data.items():
                ref_value = ref.get(key) if ref else None
                if isinstance(value, dict):
                    count += recursive_count(value, ref_value if isinstance(ref_value, dict) else None)
                elif isinstance(value, list):
                    count += sum(
                        recursive_count(v, ref_value[i] if ref_value and i < len(ref_value) else None)
                        if isinstance(v, dict) else compare_with_reference(v, ref_value[i] if ref_value else None)
                        for i, v in enumerate(value)
                    )
                else:
                    count += compare_with_reference(value, ref_value)
            return count

        return recursive_count(json_data, reference_data)
    
    async def process_awb_data(self, filename,file_bytes, reference_json=None):
        """Orchestrates AWB extraction with AutoGen AgentChat."""
        responses = {}
        
        parsers = {
            "local"  : LocalAgentParser(filename,file_bytes),
            "llm"    : LLMAgentParser(filename,file_bytes),
            "vision" : VisionAgentParser(filename,file_bytes)
        }

        for parser_name, parser in parsers.items():
            responses[parser_name] = await self.run_awb_parser_agent(parser, self.tasks[parser_name])

        best_agent = max(
            (agent for agent in responses if responses[agent]),
            key=lambda agent: self.count_filled_fields(responses[agent], reference_json),
            default=None
        )

        if best_agent is None:
            logging.error("No valid responses from agents.")
            return {"file":filename, "best_agent": None, "extracted_data": {}}

        return {
            "file":filename,
            "best_agent": parsers[best_agent].agent.name,
            "extracted_data": responses[best_agent]
        }



