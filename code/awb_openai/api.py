from pathlib import Path
import traceback
from fastapi import APIRouter, File, UploadFile, HTTPException, status

from awb_openai.main import extract_data
from logger import logger

router = APIRouter()

STATIC_FOLDER = Path("static")
STATIC_FOLDER.mkdir(parents=True, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

async def validate_input(file):
    logger.info(f"Validating input.")
    if not file or not file.filename:
        logger.error(f"Validation failed: No file provided")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")
    
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        logger.error(f"Validation failed: Invalid file type. Only PDFs and images (JPG, JPEG, PNG) are allowed.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Only PDFs and images (JPG, JPEG, PNG) are allowed."
        )

    file_bytes = await file.read()

    if not file_bytes:
        logger.error(f"Validation failed: Uploaded file is empty.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    
    logger.info("Validation successful.")
    return file_bytes

@router.post("/process_awb_data")
async def process_awb_data_api(file: UploadFile = File(...)):
    """
    Receives a file, saves it to the static folder, and processes it.
    """
    logger.info(f"Request received.")
    file_bytes = await validate_input(file)
    
    try:

        result = extract_data(file.filename,file_bytes)
        
        return result
    
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File not found after upload.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{str(e)}")