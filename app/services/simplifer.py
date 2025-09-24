import json
from pathlib import Path
from typing import Optional

class MedicalSimplifier:
    """
    Service to simplify medical terms and test results.
    Uses a JSON dictionary of medical terms -> simplified explanations.
    """

    def __init__(self, dictionary_file: str = "data/medical_terms_simplified.json"):
        self.dictionary_file = Path(dictionary_file)
        if not self.dictionary_file.exists():
            raise FileNotFoundError(f"Medical dictionary not found: {self.dictionary_file}")

        with open(self.dictionary_file, "r", encoding="utf-8") as f:
            self.simplification_dict = json.load(f)

    def simplify_term(self, term: str) -> str:
        """
        Simplify a single medical term. If not found, return the original term.
        """
        return self.simplification_dict.get(term.lower(), term)

    def simplify_text(self, text: str) -> str:
        """
        Simplify all medical terms in a text.
        """
        words = text.split()
        simplified_words = [self.simplify_term(word) for word in words]
        return " ".join(simplified_words)

    def simplify_test_result(self, test_name: str, value: str, normal_range: Optional[str] = None) -> str:
        """
        Simplify a test result into human-readable form.
        """
        simplified_name = self.simplify_term(test_name)
        explanation = f"{simplified_name} value is {value}"
        if normal_range:
            explanation += f" (normal range: {normal_range})"
        return explanation
