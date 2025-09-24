# app/routers/__init__.py

# This file makes 'routers' a package
# Optional: import all router modules for easier registration in main.py

from .upload import router as upload_router
from .chat import router as chat_router
from .reports import router as reports_router
