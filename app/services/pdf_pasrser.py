import logging
from pathlib import Path
from typing import Union
import PyPDF2
import pdfplumber
from fastapi import UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFParserService:
    """
    Service to extract text from PDF files (supports both file path & UploadFile).
    """

    def __init__(self, file: Union[str, Path, UploadFile]):
        if isinstance(file, UploadFile):
            # For FastAPI UploadFile
            self.file_path = None
            self.upload_file = file
        else:
            self.file_path = Path(file)
            self.upload_file = None
            if not self.file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {self.file_path}")

    def extract_text(self) -> str:
        """
        Extract text from PDF using PyPDF2, fallback to pdfplumber.
        """
        text = self._extract_with_pypdf2()
        if not text.strip():
            text = self._extract_with_pdfplumber()

        if not text.strip():
            raise RuntimeError("Failed to extract text from PDF with both methods.")

        return self._clean_text(text)

    def _extract_with_pypdf2(self) -> str:
        """
        Extract text using PyPDF2 (fast, basic).
        """
        text = ""
        try:
            if self.upload_file:
                # UploadFile → read bytes
                reader = PyPDF2.PdfReader(self.upload_file.file)
            else:
                with open(self.file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        return text.strip()

    def _extract_with_pdfplumber(self) -> str:
        """
        Fallback extraction using pdfplumber (better for complex PDFs).
        """
        text = ""
        try:
            if self.upload_file:
                with pdfplumber.open(self.upload_file.file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                with pdfplumber.open(self.file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        return text.strip()

    def _clean_text(self, text: str) -> str:
        """
        Normalize extracted text (remove extra whitespace).
        """
        return " ".join(text.split())

# ------------------------------
# Convenience function
# ------------------------------

def extract_text_from_pdf(file: Union[str, Path, UploadFile]) -> str:
    """
    Simple wrapper to extract text from a PDF file or FastAPI UploadFile.
    """
    parser = PDFParserService(file)
    return parser.extract_text()
