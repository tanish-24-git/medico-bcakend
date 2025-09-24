from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from app.utils.db_utils import Base

# ------------------------------
# Report Model
# ------------------------------
class ReportModel(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=True)  # Optional user identifier
    file_name = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)

# ------------------------------
# Optional: Chat History Model
# ------------------------------
class ChatHistoryModel(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=True)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# ------------------------------
# Optional: Medical Test Logs
# ------------------------------
class MedicalTestModel(Base):
    __tablename__ = "medical_tests"

    id = Column(Integer, primary_key=True, index=True)
    test_name = Column(String, nullable=False)
    description = Column(Text)
    normal_range = Column(String)
    high_value_may_indicate = Column(String)
    low_value_may_indicate = Column(String)
    upload_time = Column(DateTime, default=datetime.utcnow)
