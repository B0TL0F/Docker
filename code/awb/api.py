from pathlib import Path
import traceback
from fastapi import APIRouter, File, UploadFile, HTTPException, status

from awb.parser import OCRParser

router = APIRouter()

STATIC_FOLDER = Path("static")
STATIC_FOLDER.mkdir(parents=True, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

async def validate_input(file):
    if not file or not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")
    
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Only PDFs and images (JPG, JPEG, PNG) are allowed."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    
    return file_bytes

@router.post("/process_awb_data")
async def process_awb_data_api(file: UploadFile = File(...)):
    """
    Receives a file, saves it to the static folder, and processes it.
    """
    file_bytes = await validate_input(file)
    
    try:
        OCR_PARSER = OCRParser()

        result = await OCR_PARSER.process_awb_data(file.filename,file_bytes)
        
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File not found after upload.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {str(e)}")