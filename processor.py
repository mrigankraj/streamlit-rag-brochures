# processor.py
from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict

def extract_text_pages(pdf_path: str) -> List[str]:
    """
    Returns a list of page texts for the given PDF path.
    """
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return pages

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Simple character-based sliding-window chunker.
    Returns list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - chunk_overlap)
    return chunks

def pdf_to_chunks(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    Loads PDF, splits pages into chunks and returns list of dicts:
    { 'text': ..., 'metadata': {'source_file': filename, 'page': page_number} }
    """
    pages = extract_text_pages(pdf_path)
    results = []
    filename = Path(pdf_path).name
    for idx, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c in page_chunks:
            results.append({
                "text": c,
                "metadata": {
                    "source_file": filename,
                    "page": idx
                }
            })
    return results
