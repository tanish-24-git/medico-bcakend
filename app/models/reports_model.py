from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# ------------------------------
# Response model for a single report
# ------------------------------
class ReportResponse(BaseModel):
    id: int
    user_id: Optional[str] = None
    file_name: str
    upload_time: datetime

    class Config:
        orm_mode = True  # Allows SQLAlchemy models to work directly

# ------------------------------
# Request model for uploading a report
# ------------------------------
class ReportUploadRequest(BaseModel):
    user_id: Optional[str] = None
