from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
import faiss
import numpy as np

from .config import UPLOAD_DIR, FAISS_INDEX_PATH, METADATA_PATH
from .chatbot import get_disease_info, simplify_terms, analyze_report

app = FastAPI(
    title="SHIVAAI - AI Public Health Chatbot",
    description="Upload reports, get disease info, simplified medical terms, symptoms, and precautions",
    version="1.0.0"
)

# Allow CORS for frontend (Next.js on port 3000 and wildcard for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # Explicitly allow Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Load FAISS index and metadata
# ----------------------------
index = None
metadata = []

if Path(FAISS_INDEX_PATH).exists():
    print("Loading FAISS index...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    # Load metadata
    if Path(METADATA_PATH).exists():
        import json
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} metadata entries.")
    else:
        print("Warning: Metadata file not found.")
else:
    print("Warning: FAISS index not found. Run 'python scripts/build_rag_index.py' first.")

@app.get("/")
def home():
    return {"message": "Welcome to SHIVAAI Chatbot API"}

# ----------------------------
# Upload and Analyze Report
# ----------------------------
@app.post("/upload-report/")
async def upload_report(file: UploadFile = File(...)):
    """Endpoint to upload medical reports (PDF/Images)"""
    file_location = str(Path(UPLOAD_DIR) / file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    analysis = analyze_report(file_location)
    
    return {"filename": file.filename, "status": "uploaded successfully", "analysis": analysis}

# ----------------------------
# Ask Medical Question
# ----------------------------
@app.post("/ask-question/")
async def ask_question(question: str = Form(...)):
    """Endpoint to ask a medical question and retrieve relevant info using FAISS"""
    if index is None:
        return JSONResponse(content={"error": "FAISS index not loaded"}, status_code=500)

    # Convert question to vector
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = embedder.encode([question])
    query_vector = np.array(query_vector).astype(np.float32)

    # Search top 3 closest matches
    distances, indices = index.search(query_vector, k=3)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            results.append({
                "disease": metadata[idx]["disease"],
                "info": metadata[idx]["full_text"],
                "score": float(dist)
            })

    # Combine with general LLM response
    answer = get_disease_info(question)

    return {
        "question": question,
        "llm_answer": answer,
        "retrieved_docs": results
    }

# ----------------------------
# Simplify Medical Terms
# ----------------------------
@app.post("/simplify-term/")
async def simplify_term(term: str = Form(...)):
    """Endpoint to simplify medical term"""
    simplified = simplify_terms(term)
    return {"term": term, "simplified": simplified}

# ----------------------------
# WebSocket Endpoint for Real-Time Disease Info
# ----------------------------
@app.websocket("/ws/disease_info")
async def websocket_disease_info(websocket: WebSocket):
    """WebSocket endpoint for real-time disease info queries"""
    await websocket.accept()
    try:
        if index is None:
            await websocket.send_text("Error: FAISS index not loaded. Run 'python scripts/build_rag_index.py' first.")
            await websocket.close()
            return

        while True:
            query = await websocket.receive_text()
            # Reuse existing get_disease_info function
            answer = get_disease_info(query)
            
            # Optionally, include RAG results
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            query_vector = embedder.encode([query])
            query_vector = np.array(query_vector).astype(np.float32)
            distances, indices = index.search(query_vector, k=3)
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(metadata):
                    results.append({
                        "disease": metadata[idx]["disease"],
                        "info": metadata[idx]["full_text"],
                        "score": float(dist)
                    })

            response = {
                "question": query,
                "llm_answer": answer,
                "retrieved_docs": results
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")
        await websocket.close()
        print(f"WebSocket error: {e}")

# ----------------------------
# Start Uvicorn
# ----------------------------
if __name__ == "__main__":
    if not Path("D:\Projects\om\backend\scripts\medical_rag.index").exists():
        print("FAISS index not found. Run 'python scripts/build_rag_index.py' first.")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)