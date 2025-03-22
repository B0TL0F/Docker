import io
import os
import time
import traceback
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes

from logger import logger
from awb_openai.ocr.base_ocr import BaseOCR

# Configure Azure Computer Vision
AZURE_ENDPOINT = os.getenv("VISION_ENDPOINT")
AZURE_KEY = os.getenv("VISION_KEY")

computervision_client = None
if AZURE_ENDPOINT and AZURE_KEY:
    computervision_client = ComputerVisionClient(AZURE_ENDPOINT, CognitiveServicesCredentials(AZURE_KEY))

class AzureVisionOCR(BaseOCR):
    """OCR using Azure Computer Vision."""
    
    def extract_text(self, file_bytes):
        if computervision_client is None:
            logger.error("Azure OCR is not configured properly.")
            return None
        
        try:
            read_image = io.BytesIO(file_bytes)
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
            logger.info("OCR processing completed successfully.")
            
            return extracted_text
        except Exception as e:
            logger.error(f"Exception occurred while extracting text using azure vision ocr: {e}\n{traceback.format_exc()}")
            return None
