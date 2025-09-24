from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid
from app.services.pdf_parser import extract_text_from_pdf
from app.config import TEMP_UPLOAD_DIR

router = APIRouter(
    prefix="/upload",
    tags=["upload"]
)

# Ensure temp upload directory exists
Path(TEMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ------------------------------
# Helper functions
# ------------------------------

async def save_temp_file(file: UploadFile, path: Path):
    """
    Save uploaded file temporarily for processing.
    """
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

def clean_temp_file(path: Path):
    """
    Delete temporary file after processing.
    """
    if path.exists():
        path.unlink()

# ------------------------------
# Upload endpoint
# ------------------------------

@router.post("/report")
async def upload_medical_report(file: UploadFile = File(...)):
    """
    Upload a PDF medical report and extract text.
    Returns extracted text for further processing.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_id = str(uuid.uuid4())
    temp_file_path = Path(TEMP_UPLOAD_DIR) / f"{file_id}_{file.filename}"

    # Save uploaded file temporarily
    await save_temp_file(file, temp_file_path)

    try:
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(str(temp_file_path))

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF contains no readable text.")

        # Return extracted text along with original filename
        return {
            "status": "success",
            "file_name": file.filename,
            "extracted_text": extracted_text
        }

    finally:
        # Clean up temp file
        clean_temp_file(temp_file_path)
