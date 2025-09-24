import re
from typing import List

# ------------------------------
# Text Utilities
# ------------------------------

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, newlines, and special characters.
    """
    if not text:
        return ""
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces
    text = text.strip()
    return text

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for embedding or LLM processing.
    
    Args:
        text: Input text
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    text = clean_text(text)
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # move back by overlap for next chunk
        if start < 0:
            start = 0
        if start >= text_length:
            break

    return chunks

def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum number of characters.
    """
    text = clean_text(text)
    return text[:max_length]
