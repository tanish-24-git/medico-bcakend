import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.config import FAISS_INDEX_PATH, METADATA_PATH

# ------------------------------
# Configuration
# ------------------------------
JSON_FILE = Path("data/medical_reports_common.json")  # Path to your JSON file
INDEX_FILE = Path(FAISS_INDEX_PATH)                  # FAISS index output
METADATA_FILE = Path(METADATA_PATH)                 # Metadata output
EMBEDDING_MODEL = "all-MiniLM-L6-v2"               # SentenceTransformer model

# ------------------------------
# Load medical reports
# ------------------------------
if not JSON_FILE.exists():
    raise FileNotFoundError(f"{JSON_FILE} does not exist.")

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []
metadata = []

for item in data:
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

    documents.append(document_text)
    metadata.append(item)

print(f"Loaded {len(documents)} medical documents.")

# ------------------------------
# Create embeddings
# ------------------------------
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("Generating embeddings...")
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

# ------------------------------
# Build FAISS index
# ------------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")

# Save index
faiss.write_index(index, str(INDEX_FILE))
print(f"FAISS index saved to {INDEX_FILE}")

# Save metadata
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)
print(f"Metadata saved to {METADATA_FILE}")

print("Vector database setup completed successfully!")
