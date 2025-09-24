from pydantic import BaseModel
from typing import List, Optional

# ------------------------------
# Request Models
# ------------------------------

class ChatRequest(BaseModel):
    """
    Model for a user message to the chatbot.
    """
    user_id: Optional[str] = None  # Optional user tracking
    message: str
    history: Optional[List[str]] = []  # Previous conversation context (optional)

class ChatBulkRequest(BaseModel):
    """
    Model for sending multiple messages in batch.
    """
    user_id: Optional[str] = None
    messages: List[str]

# ------------------------------
# Response Models
# ------------------------------

class ChatResponse(BaseModel):
    """
    Model for chatbot's response.
    """
    message: str
    references: Optional[List[str]] = []  # References or related tests/diseases
    source_info: Optional[List[dict]] = []  # Metadata from FAISS retrieval

class ChatBulkResponse(BaseModel):
    """
    Response for bulk chat requests.
    """
    responses: List[ChatResponse]
