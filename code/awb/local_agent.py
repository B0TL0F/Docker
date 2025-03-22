import io
import traceback
import pytesseract  
from PIL import Image
import autogen_agentchat.agents
from pdf2image import convert_from_bytes  

from awb.common import model_client_json_data_structuring_agent, awb_schema,json_parsing_prompt

#Poppler and Tessaract are required for Local OCR processing of PDF files
poppler_path="/usr/bin"
tessaract_path="/usr/bin/tesseract"
# tessaract_path = r"C:\Program Files\Tesseract-OCR"
# poppler_path = r"C:\Program Files\poppler-24.08.0"

class LocalAgentParser():
    def __init__(self,filename:str,file_bytes):
        self.filename = filename
        self.file_bytes = file_bytes
        self.agent = self.create_agent()

    def create_agent(self):
        return autogen_agentchat.agents.AssistantAgent(
        name="awb_parser_local_agent",
        model_client=model_client_json_data_structuring_agent,
        system_message=f'''You are an AWB data extraction agent that extracts all data from AWB text and PDFs. "
                        "Use tool to get AWB document text and use it for main task."
                        "Do not loose any data provided in the input. Don't cut any data"
                        " You have a deep understanding of Airway bills data structure." 
                    
                        Fields to extract & Reponse Format JSON = 
                        {awb_schema}
                        ''',
                        
        tools=[self.extract_text_using_local_ocr],
        reflect_on_tool_use=True,
        model_client_stream=True,
        )

    def extract_text_using_local_ocr(self) -> str:
        """Tool to extract the data using local OCR"""
        try: 
            file_extension = self.filename.lower().split(".")[-1]
            
            if file_extension in ["jpg", "jpeg", "png"]:
                image = Image.open(io.BytesIO(self.file_bytes))
                pages = [image]
            elif file_extension == "pdf":
                # Convert PDF pages to images
                pages = convert_from_bytes(self.file_bytes)
            
            extracted_text_from_local = "\n".join([pytesseract.image_to_string(page) for page in pages])
            
            data = extracted_text_from_local.strip()
            return data
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}
