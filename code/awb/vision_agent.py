import io
import os
import time
import traceback

import autogen_agentchat.agents
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes


from awb.common import model_client_json_data_structuring_agent, json_parsing_prompt

endpoint = os.environ["VISION_ENDPOINT"] 
subscription_key = os.environ["VISION_KEY"] 

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

class VisionAgentParser():
    def __init__(self,filename:str,file_bytes):
        self.filename = filename
        self.file_bytes = file_bytes
        self.agent = self.create_agent()

    def create_agent(self):
        return autogen_agentchat.agents.AssistantAgent(
            name="awb_parser_ai_vision_agent",
            model_client=model_client_json_data_structuring_agent,
            system_message= f"""
                                You are an expert who can explore and understands the Airway bill(AWB) data from images and PDFs.
                                You **MUST** use the function `execute_ai_vision_based_ocr_agent` to extract the data from files. 
                                Format the data as per the provided AWB JSON Schema.
                                Do not generate responses manually. Do not hallucinate or assume any missing information. 
                                If the tool returns empty, return None.
                                Do not lose any data provided in the input. Don't cut any data.  
                              
                                {json_parsing_prompt}                   
                            """,     
            model_client_stream=True,
            tools=[self.execute_ai_vision_based_ocr_agent],
            reflect_on_tool_use=True,
            )
    
    def execute_ai_vision_based_ocr_agent(self):
        '''
        AI Computer Vision Based OCR tool to extract the data from files and images
        
        '''
        
        try:
            read_image = read_image = io.BytesIO(self.file_bytes)
            read_response = computervision_client.read_in_stream(read_image, raw=True)
            read_operation_location = read_response.headers["Operation-Location"]
            operation_id = read_operation_location.split("/")[-1]

            # Poll for OCR results
            while True:
                read_result = computervision_client.get_read_result(operation_id)
                if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                    break
                time.sleep(2)

            # Extract text from OCR results
            extracted_text = ""
            if read_result.status == OperationStatusCodes.succeeded:
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        extracted_text += line.text + "\n"
            return extracted_text

        except Exception as e:
            print(f"Error extracting text: {e}")
            error_trace = traceback.format_exc()
            print(error_trace)
            return None


 
