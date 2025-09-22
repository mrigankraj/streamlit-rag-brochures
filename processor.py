# processor.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

def load_pdf(path: str):
    path = Path(path)
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata['source_file'] = path.name
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
