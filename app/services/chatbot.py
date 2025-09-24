import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.gemini_client import GeminiClient
from app.config import FAISS_INDEX_PATH, METADATA_PATH

class ChatbotService:
    """
    Service to handle:
    - FAISS retrieval of relevant medical tests
    - Generating layman-friendly explanations via Gemini LLM
    """

    def __init__(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH, llm_client=None):
        # Load FAISS index
        try:
            self.index = faiss.read_index(index_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")

        # Load metadata
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")

        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Gemini LLM client
        self.llm_client = llm_client or GeminiClient()

    def retrieve_relevant_tests(self, query, top_k=5):
        """
        Retrieve top_k relevant medical tests from FAISS index.
        """
        if not query.strip():
            return []

        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

    def generate_prompt(self, query, retrieved_tests):
        """
        Create a prompt to send to Gemini LLM with user query
        and retrieved test info.
        """
        context_text = "\n".join([
            f"Test: {t['Test_Name']}, Normal: {t.get('Normal_Range')}, "
            f"Explanation: {t.get('Simplified_Explanation')}, "
            f"High: {t.get('High_Value_May_Indicate')}, Low: {t.get('Low_Value_May_Indicate')}"
            for t in retrieved_tests
        ])

        prompt = (
            f"User query: {query}\n"
            f"Use the following medical information to explain in simple words:\n"
            f"{context_text}\n"
            f"Answer:"
        )
        return prompt

    def get_response(self, query, top_k=5):
        """
        Retrieve relevant tests and generate layman-friendly response.
        Returns:
            response_text: str
            relevant_tests: list of dict
        """
        # 1. Retrieve relevant tests
        relevant_tests = self.retrieve_relevant_tests(query, top_k=top_k)

        if not relevant_tests:
            return "(No relevant medical tests found for this query.)", []

        # 2. Generate prompt
        prompt = self.generate_prompt(query, relevant_tests)

        # 3. Send prompt to Gemini LLM
        response_text = self.llm_client.generate(prompt)

        return response_text, relevant_tests
