# Retrieval-Augmented Generation (RAG) module for Pinecone
# Updated: Added function to store AI-generated reports

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_DIMENSION
from src.logger import setup_logger


# Load environment variables
load_dotenv()

logger = setup_logger("rag")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def init_pinecone():
    """Initialize Pinecone client and index"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [index.name for index in existing_indexes]
    
    if PINECONE_INDEX_NAME not in index_names:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        logger.info(f"Created new index: {PINECONE_INDEX_NAME}")
    
    # Connect to the index
    index = pc.Index(PINECONE_INDEX_NAME)
    logger.info(f"Connected to index: {PINECONE_INDEX_NAME}")
    return index

# Initialize index on module import
try:
    index = init_pinecone()
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    index = None

def embed_text(text: str) -> list:
    """Generate embedding for text using SentenceTransformer"""
    return embedder.encode(text).tolist()

def upsert_to_pinecone(vectors: list, ids: list, metadata: list):
    """Upsert vectors and metadata to Pinecone"""
    if index is None:
        logger.error("Pinecone index not initialized")
        return
        
    upserts = [(id, vec, meta) for id, vec, meta in zip(ids, vectors, metadata)]
    index.upsert(vectors=upserts)
    logger.info(f"Upserted {len(ids)} vectors to Pinecone.")

def get_relevant_contexts(query: str, k=3) -> list:
    """Retrieve relevant contexts from Pinecone using query embedding"""
    if index is None:
        logger.error("Pinecone index not initialized")
        return []
        
    emb = embed_text(query)
    res = index.query(vector=emb, top_k=k, include_metadata=True)
    contexts = [match['metadata']['full_text'] for match in res['matches'] if match['score'] > 0.5]
    return contexts

def store_interaction(query: str, response: str, type: str = 'query'):
    """Store user query and response in Pinecone"""
    if index is None:
        logger.error("Pinecone index not initialized")
        return
        
    emb_query = embed_text(query)
    emb_response = embed_text(response)
    metadata = {'query': query, 'response': response, 'type': type}
    upsert_to_pinecone([emb_query, emb_response], [f"q_{hash(query)}", f"r_{hash(response)}"], [metadata, metadata])

def store_report_analysis(report_text: str, analysis: str):
    """Store report and its analysis in Pinecone"""
    if index is None:
        logger.error("Pinecone index not initialized")
        return
        
    emb_report = embed_text(report_text)
    emb_analysis = embed_text(analysis)
    metadata = {'report_text': report_text, 'analysis': analysis, 'type': 'report'}
    upsert_to_pinecone([emb_report, emb_analysis], [f"rep_{hash(report_text)}", f"ana_{hash(analysis)}"], [metadata, metadata])

def store_ai_report(report_id: str, full_text: str, metadata: Dict):
    """Store AI-generated report in Pinecone for RAG retrieval"""
    if index is None:
        logger.error("Pinecone index not initialized")
        return
        
    emb = embed_text(full_text)
    upsert_to_pinecone([emb], [report_id], [{**metadata, 'full_text': full_text, 'type': 'ai_report'}])