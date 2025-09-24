from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL

# ------------------------------
# SQLAlchemy Engine
# ------------------------------
# DATABASE_URL example:
#   SQLite: "sqlite:///./shivai.db"
#   PostgreSQL: "postgresql://user:password@localhost/dbname"
#   MySQL: "mysql+pymysql://user:password@localhost/dbname"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# ------------------------------
# Session Local
# ------------------------------
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ------------------------------
# Dependency for FastAPI routes
# ------------------------------
def get_db():
    """
    FastAPI dependency to get a database session.
    Usage: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
