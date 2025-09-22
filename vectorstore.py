import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_DIR = os.environ.get("FAISS_DB_DIR", "./faiss_index")

def _get_openai_key():
    # Prefer Streamlit secrets, fallback to env
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    elif os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    else:
        raise ValueError("❌ OPENAI_API_KEY not found. Please set it in Streamlit Secrets or .env")

def create_or_get_faiss(docs):
    api_key = _get_openai_key()
    emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    db = FAISS.from_documents(docs, emb)
    db.save_local(FAISS_DIR)
    return db

def get_retriever(top_k=4):
    api_key = _get_openai_key()
    emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    db = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
    return db.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
