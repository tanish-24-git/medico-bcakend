import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shivai.db")  # Default to SQLite

# Upload directory for PDF reports
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(Path(__file__).parent.parent / "uploaded_reports"))
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# FAISS Vector Database Paths
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(Path(__file__).parent.parent.parent / "scripts" / "medical_rag.index"))
METADATA_PATH = os.getenv("METADATA_PATH", str(Path(__file__).parent.parent.parent / "scripts" / "metadata.json"))

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Other Configurations
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Debug / Environment
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
