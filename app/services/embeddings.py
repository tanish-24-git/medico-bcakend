import json
import numpy as np
import faiss
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.config import FAISS_INDEX_PATH, METADATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingsService:
    """
    Service to create FAISS embeddings for medical reports/tests.
    """

    def __init__(self, json_file_path: str, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH, model_name="all-MiniLM-L6-v2"):
        self.json_file_path = Path(json_file_path)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.metadata = []

    def load_data(self):
        """
        Load medical data from JSON and prepare documents + metadata.
        """
        if not self.json_file_path.exists():
            raise FileNotFoundError(f"{self.json_file_path} does not exist.")

        with open(self.json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if not item.get("Test_Name"):
                continue  # Skip invalid entries

            # Build text for embedding
            document_text = item.get("full_text")
            if not document_text:
                document_text = (
                    f"Test: {item['Test_Name']}. "
                    f"Normal Range: {item.get('Normal_Range', 'N/A')}. "
                    f"Explanation: {item.get('Simplified_Explanation', 'N/A')}. "
                    f"High: {item.get('High_Value_May_Indicate', 'N/A')}. "
                    f"Low: {item.get('Low_Value_May_Indicate', 'N/A')}."
                )

            self.documents.append(document_text)
            self.metadata.append(item)

        logger.info(f"Loaded {len(self.documents)} documents.")

    def create_embeddings(self):
        """
        Generate embeddings for all documents using SentenceTransformer.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_data() first.")

        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def build_faiss_index(self, embeddings):
        """
        Create FAISS index and save it along with metadata.
        """
        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))
        logger.info(f"FAISS index saved at {self.index_path}")

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)
        logger.info(f"Metadata saved at {self.metadata_path}")

    def run(self):
        """
        Run the full embedding creation process.
        """
        self.load_data()
        embeddings = self.create_embeddings()
        self.build_faiss_index(embeddings)
        logger.info("Embedding process completed successfully.")
