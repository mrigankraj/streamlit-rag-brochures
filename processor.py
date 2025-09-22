from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader
from pathlib import Path

def load_pdf(path: str):
    path = Path(path)
    try:
        loader = UnstructuredPDFLoader(str(path))
        docs = loader.load()
        for d in docs: d.metadata['source_file'] = path.name
        return docs
    except Exception:
        reader = PdfReader(str(path))
        texts = [page.extract_text() or "" for page in reader.pages]
        full = "\n\n".join(texts)
        return [Document(page_content=full, metadata={'source_file': path.name})]

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
