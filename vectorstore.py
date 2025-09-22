# vectorstore.py
import os
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# Configurable paths (from env or defaults)
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./faiss_index.index")
FAISS_DOCS_PATH = os.environ.get("FAISS_DOCS_PATH", "./faiss_docs.pkl")
SENTENCE_MODEL = os.environ.get("SENTENCE_MODEL", "all-MiniLM-L6-v2")

def _get_model(model_name: str = None):
    model_name = model_name or SENTENCE_MODEL
    return SentenceTransformer(model_name)

def _save_docs(docs, docs_path=FAISS_DOCS_PATH):
    """
    docs: list of dicts {'text':..., 'metadata':...}
    """
    Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)

def _load_docs(docs_path=FAISS_DOCS_PATH):
    if not Path(docs_path).exists():
        return None
    with open(docs_path, "rb") as f:
        return pickle.load(f)

def create_or_get_faiss(docs, model_name: str = None, batch_size: int = 32):
    """
    Build a FAISS index from docs (list of {'text','metadata'}).
    Persists index and docs to disk.
    """
    if not docs:
        raise ValueError("No docs provided to index")

    model = _get_model(model_name)
    texts = [d["text"] for d in docs]
    # Compute embeddings in batches to be memory-friendly
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size)

    # Normalize embeddings (for cosine similarity using inner product)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    # Save index and docs
    faiss.write_index(index, FAISS_INDEX_PATH)
    _save_docs(docs, FAISS_DOCS_PATH)
    return index

def load_faiss(model_name: str = None):
    """
    Load FAISS index and associated docs. Returns (index, docs) or (None, None) if not found.
    """
    if not Path(FAISS_INDEX_PATH).exists() or not Path(FAISS_DOCS_PATH).exists():
        return None, None
    index = faiss.read_index(FAISS_INDEX_PATH)
    docs = _load_docs(FAISS_DOCS_PATH)
    return index, docs

def search(query: str, top_k: int = 5, model_name: str = None):
    """
    Search the FAISS index for the query and return top_k hits with similarity scores.
    Returns a list of dicts: {'text','metadata','score'}
    """
    index, docs = load_faiss(model_name)
    if index is None or docs is None:
        return []

    model = _get_model(model_name)
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)  # D = similarity scores, I = indices
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(docs):
            continue
        results.append({
            "text": docs[idx]["text"],
            "metadata": docs[idx]["metadata"],
            "score": float(score)
        })
    return results
