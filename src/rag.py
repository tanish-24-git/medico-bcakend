import pinecone
from sentence_transformers import SentenceTransformer
import json
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_DIMENSION
from src.logger import setup_logger

logger = setup_logger("rag")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def init_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")  # Adjust environment
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=PINECONE_DIMENSION, metric='cosine')
    return pinecone.Index(PINECONE_INDEX_NAME)

index = init_pinecone()

def embed_text(text: str) -> list:
    return embedder.encode(text).tolist()

def upsert_to_pinecone(vectors: list, ids: list, metadata: list):
    upserts = [(id, vec, meta) for id, vec, meta in zip(ids, vectors, metadata)]
    index.upsert(vectors=upserts)
    logger.info(f"Upserted {len(ids)} vectors to Pinecone.")

def get_relevant_contexts(query: str, k=3) -> list:
    emb = embed_text(query)
    res = index.query(emb, top_k=k, include_metadata=True)
    contexts = [match['metadata']['full_text'] for match in res['matches'] if match['score'] > 0.5]  # Threshold for relevance
    return contexts

def store_interaction(query: str, response: str, type: str = 'query'):
    emb_query = embed_text(query)
    emb_response = embed_text(response)
    metadata = {'query': query, 'response': response, 'type': type}
    upsert_to_pinecone([emb_query, emb_response], [f"q_{hash(query)}", f"r_{hash(response)}"], [metadata, metadata])

def store_report_analysis(report_text: str, analysis: str):
    emb_report = embed_text(report_text)
    emb_analysis = embed_text(analysis)
    metadata = {'report_text': report_text, 'analysis': analysis, 'type': 'report'}
    upsert_to_pinecone([emb_report, emb_analysis], [f"rep_{hash(report_text)}", f"ana_{hash(analysis)}"], [metadata, metadata])