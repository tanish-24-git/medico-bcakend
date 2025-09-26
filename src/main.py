# FastAPI application: Main entry point
# Updated: New endpoints/WS for calls, prescriptions, AI

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.chatbot_service import get_disease_info, simplify_terms, chatbot
from src.report_analyzer import analyze_report
from src.logger import setup_logger
from src.rag import get_relevant_contexts
from src.firebase_service import verify_auth_token, create_video_session, update_video_session
from src.video_call_service import process_recording
from src.prescription_service import add_prescription
import uuid
from typing import Dict

class QuestionRequest(BaseModel):
    question: str

class CreateSessionRequest(BaseModel):
    patient_id: str
    doctor_id: str
    id_token: str  # Firebase ID token

class AddPrescriptionRequest(BaseModel):
    session_id: str
    patient_id: str
    doctor_id: str
    medication: str
    dosage: str
    instructions: str
    id_token: str

class UploadRecordingRequest(BaseModel):
    session_id: str
    id_token: str

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

@app.post("/create-video-session/")
async def create_video_session(request: CreateSessionRequest):
    """Create video session metadata in Firestore."""
    try:
        decoded_token = verify_auth_token(request.id_token)
        session_id = str(uuid.uuid4())
        data = {
            'date': firestore.SERVER_TIMESTAMP,
            'participants': [{'role': 'patient', 'uid': request.patient_id}, {'role': 'doctor', 'uid': request.doctor_id}],
            'recording_url': None,
            'metadata': {'duration': 0}  # Update later
        }
        create_video_session(session_id, data)
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=401, detail="Auth failed or error")

@app.post("/add-prescription/")
async def api_add_prescription(request: AddPrescriptionRequest):
    """Add prescription during/after call."""
    try:
        decoded_token = verify_auth_token(request.id_token)
        if decoded_token['uid'] != request.doctor_id:
            raise ValueError("Only doctor can add prescription")
        prescription_id = add_prescription(
            request.session_id, request.patient_id, request.doctor_id,
            request.medication, request.dosage, request.instructions
        )
        return {"prescription_id": prescription_id}
    except Exception as e:
        logger.error(f"Error adding prescription: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-recording/")
async def upload_recording(file: UploadFile = File(...), session_id: str = Form(...), id_token: str = Form(...)):
    """Upload 30s recording, store in Cloudinary, trigger AI processing."""
    try:
        decoded_token = verify_auth_token(id_token)
        file_path = f"/tmp/{file.filename}"  # Temp save
        with open(file_path, "wb") as f:
            f.write(await file.read())
        destination = f"recordings/{session_id}/{file.filename}"
        url = upload_to_storage(file_path, destination)
        update_video_session(session_id, {'recording_url': url})
        # Trigger AI processing synchronously for prototype
        process_recording(url, session_id)
        return {"status": "uploaded and processing"}
    except Exception as e:
        logger.error(f"Error uploading recording: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/signaling/{session_id}")
async def websocket_signaling(websocket: WebSocket, session_id: str):
    """WebSocket for WebRTC signaling (offer/answer/ICE)."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # Broadcast to other participant (for simplicity, assume 2 participants; in prod, use rooms/channels)
            # Here, just echo for prototype; implement proper signaling logic in frontend/backend.
            await websocket.send_json({"type": "signal", "data": data})
    except WebSocketDisconnect:
        logger.info("Signaling WebSocket disconnected")

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