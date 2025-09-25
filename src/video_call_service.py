# Handles video call post-processing: audio extraction, STT, AI report generation
# New file for processing call recordings

import whisper
from pydub import AudioSegment
import requests
import io
import uuid
import json
from src.logger import setup_logger
from src.chatbot_service import chatbot
from src.rag import store_ai_report
from src.firebase_service import db, create_report, upload_to_storage, get_linked_prescription
from src.report_analyzer import perform_comprehensive_analysis

logger = setup_logger("video_call")

# Load Whisper tiny model for free-tier STT (CPU-friendly)
model = whisper.load_model("tiny")

def process_recording(video_url: str, session_id: str):
    """Process video recording: extract audio, transcribe, generate AI report, store"""
    try:
        # Download video from Firebase Storage
        response = requests.get(video_url)
        video_content = io.BytesIO(response.content)
        
        # Extract audio (assume MP4 input)
        audio = AudioSegment.from_file(video_content, format="mp4")
        audio_path = f"/tmp/audio_{session_id}.wav"
        audio.export(audio_path, format="wav")
        
        # Transcribe audio (limited to 30s for free-tier)
        transcript = model.transcribe(audio_path)["text"]
        logger.info(f"Transcribed: {transcript[:100]}...")  # Truncate log
        
        # Fetch linked prescription
        session_ref = db.collection('video_sessions').document(session_id)
        prescription = get_linked_prescription(session_ref)
        
        # Generate AI report
        report_content = chatbot.generate_response(
            user_input="Generate a structured medical report from this call transcript.",
            transcript=transcript,
            prescription=prescription
        )
        
        # Enhance with comprehensive analysis
        analysis = perform_comprehensive_analysis(report_content, transcript, prescription)
        
        # Store in Firestore
        report_id = str(uuid.uuid4())
        session = db.collection('video_sessions').document(session_id).get().to_dict()
        patient_ref = next(p['uid'] for p in session['participants'] if p['role'] == 'patient')
        doctor_ref = next(p['uid'] for p in session['participants'] if p['role'] == 'doctor')
        data = {
            'type': 'ai_generated',
            'date': firestore.SERVER_TIMESTAMP,
            'content': {
                'medications': prescription.get('medication') if prescription else '',
                'observations': analysis
            },
            'patient_ref': db.collection('patients').document(patient_ref),
            'doctor_ref': db.collection('doctors').document(doctor_ref),
            'session_ref': session_ref,
            'file_url': None  # Update after storage
        }
        create_report(report_id, data)
        
        # Store in Pinecone for RAG
        store_ai_report(report_id, analysis, {'source': 'ai_call_report'})
        
        # Save report as JSON to Storage
        json_path = f"/tmp/report_{report_id}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)
        destination = f"reports/{report_id}.json"
        url = upload_to_storage(json_path, destination)
        db.collection('reports').document(report_id).update({'file_url': url})
        
        logger.info(f"AI report generated and stored: {report_id}")
        
    except Exception as e:
        logger.error(f"Error processing recording: {e}")