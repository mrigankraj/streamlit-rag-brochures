import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_DIR = os.environ.get("FAISS_DB_DIR", "./faiss_db")

def create_or_get_faiss(docs):
    """Create FAISS index from docs and save locally"""
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(docs, emb)
    db.save_local(FAISS_DIR)
    return db

def get_retriever(top_k=4):
    """Load FAISS index from disk and return retriever"""
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k},
    )
