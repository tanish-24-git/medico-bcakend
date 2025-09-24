from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json
from pydantic import BaseModel
from app.config import UPLOAD_DIR

router = APIRouter(
    prefix="/reports",
    tags=["reports"]
)

# Ensure upload directory exists
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
METADATA_FILE = Path(UPLOAD_DIR) / "reports_metadata.json"

# Initialize metadata file if it does not exist
if not METADATA_FILE.exists():
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# ------------------------------
# Pydantic models
# ------------------------------

class ReportInfo(BaseModel):
    report_id: str
    file_name: str
    extracted_text: str

# ------------------------------
# Helper functions
# ------------------------------

def load_report_metadata():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_report_metadata(metadata):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

# ------------------------------
# API Endpoints
# ------------------------------

@router.get("/", response_model=list[ReportInfo])
async def list_reports():
    """
    List all uploaded reports with metadata.
    """
    return load_report_metadata()

@router.get("/download")
async def download_report(report_id: str = Query(..., description="ID of the report to download")):
    """
    Download a specific PDF report by its report_id.
    """
    metadata = load_report_metadata()
    report = next((r for r in metadata if r["report_id"] == report_id), None)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    file_path = Path(UPLOAD_DIR) / report["file_name"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on server")

    return FileResponse(path=str(file_path), filename=report["file_name"], media_type="application/pdf")

@router.post("/save")
async def save_report(file_name: str, extracted_text: str):
    """
    Save a new uploaded report's metadata.
    Call this after successfully uploading and extracting text from a PDF.
    """
    metadata = load_report_metadata()
    report_id = str(len(metadata) + 1)  # simple incremental ID

    new_report = {
        "report_id": report_id,
        "file_name": file_name,
        "extracted_text": extracted_text
    }

    metadata.append(new_report)
    save_report_metadata(metadata)

    return JSONResponse({"status": "success", "report_id": report_id})
