import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.utils.text_utils import clean_text, split_text_into_chunks
from app.config import FAISS_INDEX_PATH, METADATA_PATH

# ------------------------------
# Configuration
# ------------------------------
NEW_JSON_FILE = Path("data/new_medical_reports.json")  # New reports
INDEX_FILE = Path(FAISS_INDEX_PATH)
METADATA_FILE = Path(METADATA_PATH)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ------------------------------
# Load new medical reports
# ------------------------------
if not NEW_JSON_FILE.exists():
    raise FileNotFoundError(f"{NEW_JSON_FILE} does not exist.")

with open(NEW_JSON_FILE, "r", encoding="utf-8") as f:
    new_data = json.load(f)

new_documents = []
new_metadata = []

for item in new_data:
    if not item.get("Test_Name"):
        continue

    document_text = item.get("full_text")
    if not document_text:
        document_text = (
            f"Test: {item['Test_Name']}. "
            f"Normal Range: {item.get('Normal_Range', 'N/A')}. "
            f"Explanation: {item.get('Simplified_Explanation', 'N/A')}. "
            f"High: {item.get('High_Value_May_Indicate', 'N/A')}. "
            f"Low: {item.get('Low_Value_May_Indicate', 'N/A')}."
        )

    # Clean and chunk text
    document_text = clean_text(document_text)
    chunks = split_text_into_chunks(document_text, chunk_size=500, overlap=50)

    new_documents.extend(chunks)
    for _ in chunks:
        new_metadata.append(item)

print(f"Loaded {len(new_documents)} new text chunks from {len(new_data)} new reports.")

# ------------------------------
# Load existing FAISS index and metadata
# ------------------------------
if not INDEX_FILE.exists() or not METADATA_FILE.exists():
    raise FileNotFoundError("FAISS index or metadata not found. Run setup_vector_db.py first.")

index = faiss.read_index(str(INDEX_FILE))
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    existing_metadata = json.load(f)

# ------------------------------
# Generate embeddings for new documents
# ------------------------------
model = SentenceTransformer(EMBEDDING_MODEL)
print("Generating embeddings for new documents...")
new_embeddings = model.encode(new_documents, show_progress_bar=True)
new_embeddings = np.array(new_embeddings).astype("float32")
print(f"New embeddings shape: {new_embeddings.shape}")

# ------------------------------
# Add new vectors to FAISS index
# ------------------------------
index.add(new_embeddings)
faiss.write_index(index, str(INDEX_FILE))
print(f"FAISS index updated. Total vectors: {index.ntotal}")

# ------------------------------
# Update metadata
# ------------------------------
existing_metadata.extend(new_metadata)
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(existing_metadata, f, ensure_ascii=False, indent=4)
print(f"Metadata updated. Total entries: {len(existing_metadata)}")

print("FAISS vector database successfully updated with new medical reports!")
