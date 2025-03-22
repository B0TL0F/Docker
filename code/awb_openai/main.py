import os
import json
import traceback

from logger import logger
from awb_openai.llm import extractions_llm
from awb_openai.ocr.main import OCRFactory


with open(os.path.join("awb_openai","schemas","json_schema.txt"), "r", encoding="utf-8") as json_schema:
    awb_schema = json_schema.read()

instructions = '''
                Dont add any comments in json output. 
                Dont use markdown. 
                Dont add new keys and objects in json. 
                Try to capture accurate and logical details.
                Keep the json key order same.
                '''

def extract_data(filename, file_bytes):
    try:
    
        extracted_text = OCRFactory.call_ocr('azure',file_bytes,filename)
        
        if extracted_text == None:
            raise Exception("OCR extraction returned None.")
            
        output = extractions_llm(extracted_text,'o1',awb_schema,instructions)
        
        return json.loads(output)
    
    except Exception as e:
        logger.error(f"Exception occurred while extracting AWB {filename}: {e}\n{traceback.format_exc()}")
        raise Exception("Sorry we are unable to provide service. Please try again later.")



