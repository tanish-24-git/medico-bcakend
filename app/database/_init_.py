# app/database/__init__.py

# This file makes 'database' a package
# Optional: import the DB connection and models for easy access

from .db import database, get_db_session
from .models import *  # Import all ORM models
