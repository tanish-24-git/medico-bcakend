# Handles Firebase initialization and CRUD operations for entities and storage
# New file for managing Firebase Authentication, Firestore, and Storage

import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
from src.config import FIREBASE_SERVICE_ACCOUNT_PATH
from src.logger import setup_logger
from typing import Dict, Any, Optional, List

logger = setup_logger("firebase_service")

# Initialize Firebase app (called once)
cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()  # Firestore client
bucket = storage.bucket()  # Default Storage bucket

def verify_auth_token(id_token: str) -> Dict:
    """Verify Firebase ID token from frontend for authentication"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        logger.info(f"Verified user: {decoded_token['uid']}")
        return decoded_token
    except Exception as e:
        logger.error(f"Auth verification failed: {e}")
        raise ValueError("Invalid auth token")

def create_user(uid: str, data: Dict) -> None:
    """Create user document in patients or doctors collection"""
    collection = 'patients' if data.get('role') == 'patient' else 'doctors'
    db.collection(collection).document(uid).set(data)
    logger.info(f"Created {collection} user: {uid}")

def get_user(uid: str, role: str) -> Optional[Dict]:
    """Get user document by UID and role"""
    collection = 'patients' if role == 'patient' else 'doctors'
    doc = db.collection(collection).document(uid).get()
    if doc.exists:
        return doc.to_dict()
    return None

def create_hospital(hospital_id: str = "1234", data: Dict = {}) -> None:
    """Create hospital document with prototype ID"""
    default_data = {
        'name': 'Prototype Hospital',
        'location': 'Default',
        'email': 'hospital@example.com',
        'employees': [],
        'video_sessions': [],
        'ai_reports': [],
        'prescriptions': [],
        'patients': []
    }
    db.collection('hospitals').document(hospital_id).set({**default_data, **data})
    logger.info(f"Created hospital: {hospital_id}")

def add_to_hospital(hospital_id: str, field: str, value: Any) -> None:
    """Add to hospital arrays (e.g., employees, patients, prescriptions)"""
    db.collection('hospitals').document(hospital_id).update({field: firestore.ArrayUnion([value])})

def create_video_session(session_id: str, data: Dict) -> None:
    """Create video session metadata in Firestore"""
    db.collection('video_sessions').document(session_id).set(data)
    logger.info(f"Created video session: {session_id}")

def update_video_session(session_id: str, updates: Dict) -> None:
    """Update video session (e.g., add recording URL)"""
    db.collection('video_sessions').document(session_id).update(updates)

def create_report(report_id: str, data: Dict) -> None:
    """Create AI or doctor-generated report in Firestore"""
    db.collection('reports').document(report_id).set(data)
    logger.info(f"Created report: {report_id}")

def create_prescription(prescription_id: str, data: Dict) -> str:
    """Create prescription and link to patient, doctor, session, hospital"""
    db.collection('prescriptions').document(prescription_id).set(data)
    # Link to entities
    patient_ref = data['patient_ref']
    doctor_ref = data['doctor_ref']
    session_ref = data['session_ref']
    hospital_id = data.get('hospital_id', '1234')
    db.collection('patients').document(patient_ref.id).update({
        'prescriptions': firestore.ArrayUnion([db.collection('prescriptions').document(prescription_id)])
    })
    db.collection('doctors').document(doctor_ref.id).update({
        'prescriptions': firestore.ArrayUnion([db.collection('prescriptions').document(prescription_id)])
    })
    db.collection('video_sessions').document(session_ref.id).update({
        'prescription_ref': db.collection('prescriptions').document(prescription_id)
    })
    add_to_hospital(hospital_id, 'prescriptions', db.collection('prescriptions').document(prescription_id))
    logger.info(f"Created and linked prescription: {prescription_id}")
    return prescription_id

def upload_to_storage(file_path: str, destination: str) -> str:
    """Upload file (video/report) to Firebase Storage, return public URL"""
    blob = bucket.blob(destination)
    blob.upload_from_filename(file_path)
    blob.make_public()  # For prototype; secure in production
    logger.info(f"Uploaded to storage: {destination}")
    return blob.public_url

def get_linked_prescription(session_ref) -> Optional[Dict]:
    """Fetch prescription linked to a session for AI report generation"""
    session = db.collection('video_sessions').document(session_ref.id).get().to_dict()
    if 'prescription_ref' in session:
        return db.document(session['prescription_ref'].path).get().to_dict()
    return None