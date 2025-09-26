# Configuration module: Loads environment variables
# Updated: Added Cloudinary config

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# API keys and settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", 384))
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")  # Path to Firebase service account JSON
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Debug and environment settings
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")