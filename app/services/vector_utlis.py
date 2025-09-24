import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.config import FAISS_INDEX_PATH, METADATA_PATH

class VectorUtilsService:
    """
    Service class for interacting with FAISS vector database.
    """

    def __init__(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        # Load FAISS index
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        # Load metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def query(self, text: str, top_k=5):
        """
        Search for top_k relevant vectors in FAISS based on input text.
        Returns list of metadata items.
        """
        if not text.strip():
            return []

        vector = self.model.encode([text]).astype("float32")
        distances, indices = self.index.search(vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

    def add_new_vectors(self, texts: list, metadata_list: list):
        """
        Optional: Add new vectors to the FAISS index.
        """
        if len(texts) != len(metadata_list):
            raise ValueError("texts and metadata_list must have the same length.")

        new_vectors = self.model.encode(texts).astype("float32")
        self.index.add(new_vectors)

        # Update metadata
        self.metadata.extend(metadata_list)

        # Save index and metadata
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)
        print(f"Added {len(texts)} new vectors to FAISS index.")
