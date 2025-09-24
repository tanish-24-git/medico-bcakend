import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH, METADATA_PATH

# ----------------------------------------
# 1. Load disease data
# ----------------------------------------
data_path = Path(__file__).parent.parent / 'app' / 'data' / 'medical_reports_common.json'

with open(data_path, 'r', encoding='utf-8') as f:
    diseases = json.load(f)

if not diseases:
    raise ValueError("The medical_reports_common.json file is empty. Please add data.")

print(f"Loaded {len(diseases)} records from {data_path}")

# ----------------------------------------
# 2. Prepare chunks for embeddings
# ----------------------------------------
# Use 'full_text' if available, otherwise fallback to 'description'
chunks = [(d.get('full_text') or d.get('description') or '').strip() for d in diseases]

# Filter out empty strings to prevent embedding errors
valid_data = [(i, d, text) for i, (d, text) in enumerate(zip(diseases, chunks)) if text]

if not valid_data:
    raise ValueError("No valid 'full_text' or 'description' found in the dataset.")

print(f"Found {len(valid_data)} valid records for embedding")

# Extract clean texts for embedding
clean_texts = [item[2] for item in valid_data]

# ----------------------------------------
# 3. Generate embeddings
# ----------------------------------------
print("Generating embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(clean_texts)

print(f"Embeddings shape: {embeddings.shape}")

# ----------------------------------------
# 4. Build FAISS index
# ----------------------------------------
print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)

# Convert embeddings to float32 for FAISS
index.add(np.array(embeddings).astype(np.float32))

# Save FAISS index to disk
faiss.write_index(index, str(FAISS_INDEX_PATH))
print(f"FAISS index saved at {FAISS_INDEX_PATH}")

# ----------------------------------------
# 5. Save metadata
# ----------------------------------------
metadata = [
    {
        'id': idx,
        'disease': d.get('disease', 'Unknown'),
        'full_text': text
    }
    for idx, d, text in valid_data
]

with open(METADATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Metadata saved at {METADATA_PATH}")
print("FAISS index build complete ✅")
