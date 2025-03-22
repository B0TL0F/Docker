import io
import traceback
import pytesseract  
from PIL import Image
from pdf2image import convert_from_bytes  

from logger import logger
from awb_openai.ocr.base_ocr import BaseOCR

class TesseractOCR(BaseOCR):
    """OCR using Tesseract."""

    def extract_text(self, file_bytes, filename):
        try: 
            file_extension = filename.lower().split(".")[-1]
            
            if file_extension in ["jpg", "jpeg", "png"]:
                image = Image.open(io.BytesIO(file_bytes))
                pages = [image]
            elif file_extension == "pdf":
                pages = convert_from_bytes(file_bytes)
            
            extracted_text_from_local = "\n".join([pytesseract.image_to_string(page) for page in pages])
            
            data = extracted_text_from_local.strip()
            logger.info("OCR processing completed successfully.")
            return data
        except Exception as e:
            logger.error(f"Exception occurred while extracting text using tessseract ocr: {e}\n{traceback.format_exc()}")
            return None
