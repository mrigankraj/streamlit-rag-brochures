import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

CHROMA_DIR = os.environ.get('CHROMA_DB_DIR', './chroma_db')

def create_or_get_chroma(docs, persist=True):
    emb = OpenAIEmbeddings()
    db = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=CHROMA_DIR)
    if persist: db.persist()
    return db

def get_retriever(top_k=4):
    emb = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
    return db.as_retriever(search_type='mmr', search_kwargs={'k': top_k})
