# Handles prescription creation and linking to entities
# New file for managing doctor prescriptions

from src.firebase_service import db, create_prescription
from src.logger import setup_logger
import uuid

logger = setup_logger("prescription")

def add_prescription(session_id: str, patient_id: str, doctor_id: str, medication: str, dosage: str, instructions: str) -> str:
    """Create prescription and link to patient, doctor, session, and hospital"""
    prescription_id = str(uuid.uuid4())
    patient_ref = db.collection('patients').document(patient_id)
    doctor_ref = db.collection('doctors').document(doctor_id)
    session_ref = db.collection('video_sessions').document(session_id)
    data = {
        'date': firestore.SERVER_TIMESTAMP,
        'medication': medication,
        'dosage': dosage,
        'instructions': instructions,
        'doctor_ref': doctor_ref,
        'patient_ref': patient_ref,
        'session_ref': session_ref,
        'hospital_id': '1234'  # Prototype fixed ID
    }
    return create_prescription(prescription_id, data)