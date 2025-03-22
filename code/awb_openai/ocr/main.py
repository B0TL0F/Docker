from awb_openai.ocr.tesseract import TesseractOCR
from awb_openai.ocr.azure_vision import AzureVisionOCR

from logger import logger

class OCRFactory:
    """Factory to create OCR objects."""
    
    @staticmethod
    def call_ocr(ocr_type,file_bytes, filename):
        logger.info(f"OCR request initiated: {ocr_type}")
        if ocr_type == "tesseract":
            return TesseractOCR().extract_text(file_bytes,filename)
        elif ocr_type == "azure":
            return AzureVisionOCR().extract_text(file_bytes)
        else:
            print(f"OCR {ocr_type} unsupported")
            raise ValueError(f"Unsupported OCR type: {ocr_type}. Supported = ['tesseract','azure']")