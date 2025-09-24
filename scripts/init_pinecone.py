import sys
import os
# Use raw string for absolute path
sys.path.append(r'D:\Projects\om\backend')
from src.rag import embed_text, upsert_to_pinecone
from src.logger import setup_logger
import json

logger = setup_logger("init_pinecone")

def load_initial_data():
    # Use raw string for file path
    with open(r'D:\Projects\om\backend\data\medical_reports_common.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Debug: Print the first entry to see the actual structure
    if data:
        logger.info(f"First entry keys: {list(data[0].keys())}")
        logger.info(f"Sample entry: {data[0]}")
    
    texts = []
    for i, entry in enumerate(data):
        # Try different possible field names for the text content
        text = ""
        if 'fulltext' in entry:
            text = entry['fulltext']
        elif 'full_text' in entry:
            text = entry['full_text']
        elif 'simplified_explanation' in entry:
            text = entry['simplified_explanation']
        elif 'disease' in entry:
            # Combine disease name with explanation as fallback
            disease = entry.get('disease', '')
            explanation = entry.get('simplified_explanation', '')
            text = f"Disease: {disease}. Explanation: {explanation}"
        
        if text and text.strip():
            texts.append(text)
        else:
            logger.warning(f"No text content found for entry {i}: {entry}")
    
    if not texts:
        logger.error("No text content found in any entries!")
        return
    
    logger.info(f"Processing {len(texts)} entries")
    
    embeddings = [embed_text(text) for text in texts]
    ids = [f"med_{i}" for i in range(len(texts))]
    metadata = [{'full_text': text, 'type': 'medical_knowledge'} for text in texts]
    
    upsert_to_pinecone(embeddings, ids, metadata)
    logger.info(f"Successfully loaded {len(texts)} medical entries into Pinecone.")

if __name__ == "__main__":
    load_initial_data()
