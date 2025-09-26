# Handles Firebase initialization and CRUD operations for entities (Firestore only; Storage swapped to Cloudinary)
# Updated: Removed Firebase Storage; added Cloudinary config/upload

import firebase_admin
from firebase_admin import credentials, firestore, auth
from src.config import FIREBASE_SERVICE_ACCOUNT_PATH, CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET
from src.logger import setup_logger
from typing import Dict, Any, Optional, List
import cloudinary
import cloudinary.uploader

logger = setup_logger("firebase_service")

# Initialize Firebase app (Firestore only)
cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()  # Firestore client

# Initialize Cloudinary for storage
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True  # Use HTTPS
)
logger.info("Cloudinary initialized")

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
    """Create hospital document (prototype fixed ID)."""
    default_data = {'name': 'Prototype Hospital', 'location': 'Default', 'email': 'hospital@example.com', 'employees': [], 'video_sessions': [], 'ai_reports': [], 'prescriptions': [], 'patients': []}
    db.collection('hospitals').document(hospital_id).set({**default_data, **data})
    logger.info(f"Created hospital: {hospital_id}")

def add_to_hospital(hospital_id: str, field: str, value: Any) -> None:
    """Add to hospital arrays (e.g., employees, patients)."""
    db.collection('hospitals').document(hospital_id).update({field: firestore.ArrayUnion([value])})

def create_video_session(session_id: str, data: Dict) -> None:
    """Create video session metadata."""
    db.collection('video_sessions').document(session_id).set(data)
    logger.info(f"Created video session: {session_id}")

def update_video_session(session_id: str, updates: Dict) -> None:
    """Update session (e.g., add recording_url)."""
    db.collection('video_sessions').document(session_id).update(updates)

def create_report(report_id: str, data: Dict) -> None:
    """Create AI/doctor report."""
    db.collection('reports').document(report_id).set(data)
    logger.info(f"Created report: {report_id}")

def create_prescription(prescription_id: str, data: Dict) -> str:
    """Create prescription, link to entities."""
    db.collection('prescriptions').document(prescription_id).set(data)
    # Link to patient, doctor, session, hospital
    patient_ref = data['patient_ref']
    doctor_ref = data['doctor_ref']
    session_ref = data['session_ref']
    hospital_id = data.get('hospital_id', '1234')
    db.collection('patients').document(patient_ref.id).update({'prescriptions': firestore.ArrayUnion([db.collection('prescriptions').document(prescription_id)])})
    db.collection('doctors').document(doctor_ref.id).update({'prescriptions': firestore.ArrayUnion([db.collection('prescriptions').document(prescription_id)])})
    db.collection('video_sessions').document(session_ref.id).update({'prescription_ref': db.collection('prescriptions').document(prescription_id)})
    add_to_hospital(hospital_id, 'prescriptions', db.collection('prescriptions').document(prescription_id))
    logger.info(f"Created and linked prescription: {prescription_id}")
    return prescription_id

def upload_to_storage(file_path: str, destination: str) -> str:
    """Upload file (video/report) to Cloudinary, return public URL."""
    try:
        response = cloudinary.uploader.upload(
            file_path,
            folder=destination,
            resource_type="auto"  # Auto-detect video/image/raw
        )
        url = response['secure_url']
        logger.info(f"Uploaded to Cloudinary: {destination} - URL: {url}")
        return url
    except Exception as e:
        logger.error(f"Cloudinary upload error: {e}")
        raise

def get_linked_prescription(session_ref) -> Optional[Dict]:
    """Fetch prescription for a session (for AI report)."""
    session = db.collection('video_sessions').document(session_ref.id).get().to_dict()
    if 'prescription_ref' in session:
        return db.document(session['prescription_ref'].path).get().to_dict()
    return None

# Other CRUD methods can be added as needed (e.g., list_sessions_for_user).