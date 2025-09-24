import sys
import os
# Add the parent directory (backend) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'D:\Projects\om\backend')))
from src.rag import embed_text, upsert_to_pinecone
from src.logger import setup_logger
import json

logger = setup_logger("init_pinecone")

def load_initial_data():
    with open('../data/medical_reports_common.json', 'r') as f:
        data = json.load(f)
    
    texts = [entry['full_text'] for entry in data]
    embeddings = [embed_text(text) for text in texts]
    ids = [f"med_{i}" for i in range(len(data))]
    metadata = [{'full_text': text, 'type': 'medical_knowledge'} for text in texts]
    
    upsert_to_pinecone(embeddings, ids, metadata)
    logger.info("Initial medical data loaded into Pinecone.")

if __name__ == "__main__":
    load_initial_data()