from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.chatbot_service import get_disease_info, simplify_terms, chatbot
from src.report_analyzer import analyze_report
from src.logger import setup_logger
from src.rag import get_relevant_contexts

class QuestionRequest(BaseModel):
    question: str

logger = setup_logger("main")

app = FastAPI(
    title="SHIVAAI - AI Public Health Chatbot",
    description="Upload reports, get disease info, simplified medical terms, symptoms, and precautions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to SHIVAAI Chatbot API"}

@app.post("/upload-report/")
async def upload_report(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        analysis = analyze_report(file_content)
        return {"filename": file.filename, "status": "analyzed successfully", "analysis": analysis}
    except Exception as e:
        logger.error(f"Error in upload_report: {str(e)}")
        return {"error": str(e)}

@app.post("/ask-question/")
async def ask_question(question: QuestionRequest):
    try:
        response = chatbot.generate_response(question.question)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return {"error": str(e)}

@app.post("/simplify-term/")
async def simplify_term(term: str = Form(...)):
    try:
        simplified = simplify_terms(term)
        return {"term": term, "simplified": simplified}
    except Exception as e:
        logger.error(f"Error in simplify_term: {str(e)}")
        return {"error": str(e)}

@app.websocket("/ws/disease_info")
async def websocket_disease_info(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            answer = get_disease_info(query)
            retrieved_docs = get_relevant_contexts(query)
            response = {
                "question": query,
                "llm_answer": answer,
                "retrieved_docs": retrieved_docs
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")
        await websocket.close()
        logger.error(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001, reload=True)
