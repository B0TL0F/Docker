import os
import json
from autogen_ext.models.openai import OpenAIChatCompletionClient

with open(os.path.join("awb","schemas","awb-schema.txt"), "r", encoding="utf-8") as json_schema:
    awb_schema = json_schema.read()

with open(os.path.join("awb","prompts","json-parsing-prompt.txt"), "r", encoding="utf-8") as json_prompt_samples:
    json_parsing_prompt = json_prompt_samples.read() 


llm_config = {"model": "gpt-4o-2024-08-06"}

model_client_json_data_structuring_agent = OpenAIChatCompletionClient(
            model=llm_config["model"],          
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "awb_schema",
                    "strict": True,
                    "schema":  json.loads(awb_schema)
                }
            }        
)