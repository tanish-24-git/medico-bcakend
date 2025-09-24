from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.chatbot import ChatbotService
from app.services.gemini_client import GeminiClient

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

# ------------------------------
# Pydantic models
# ------------------------------

class ChatRequest(BaseModel):
    user_text: str  # User's question or extracted PDF text

class TestInfo(BaseModel):
    Test_Name: str
    Normal_Range: str = None
    Simplified_Explanation: str = None
    High_Value_May_Indicate: str = None
    Low_Value_May_Indicate: str = None

class ChatResponse(BaseModel):
    response_text: str
    relevant_tests: List[TestInfo]

# ------------------------------
# Initialize Chatbot service
# ------------------------------

llm_client = GeminiClient()  # Uses API key from config
chatbot_service = ChatbotService(llm_client=llm_client)

# ------------------------------
# POST endpoint for chat
# ------------------------------

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.user_text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1️⃣ Retrieve top relevant tests and generate response
    response_text, relevant_tests = chatbot_service.get_response(query, top_k=5)

    # 2️⃣ Format relevant tests for response
    relevant_tests_list = [
        TestInfo(
            Test_Name=t.get("Test_Name"),
            Normal_Range=t.get("Normal_Range"),
            Simplified_Explanation=t.get("Simplified_Explanation"),
            High_Value_May_Indicate=t.get("High_Value_May_Indicate"),
            Low_Value_May_Indicate=t.get("Low_Value_May_Indicate")
        )
        for t in relevant_tests
    ]

    # 3️⃣ Return structured response
    return ChatResponse(
        response_text=response_text,
        relevant_tests=relevant_tests_list
    )
