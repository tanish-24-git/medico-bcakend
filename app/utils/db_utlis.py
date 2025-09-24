from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URL

# ------------------------------
# SQLAlchemy setup
# ------------------------------

# Create database engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Only for SQLite, ignore for MySQL/Postgres
)

# Session local class for DB connections
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# ------------------------------
# Dependency function for FastAPI
# ------------------------------

def get_db():
    """
    FastAPI dependency to get a database session.
    Usage:
        db = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------------------
# Utility functions
# ------------------------------

def init_db():
    """
    Create all tables in the database. Call this at startup.
    """
    from app.database.models import Base  # Import your models here
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")
