# app.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import openai
import pandas as pd

from processor import pdf_to_chunks
from vectorstore import create_or_get_faiss, search, load_faiss
from downloader import download_from_excel  # keeps your downloader workflow

load_dotenv()

st.set_page_config(page_title="Brochure RAG — Local Embeddings (sentence-transformers)", layout="wide")
st.title("📄 Brochure RAG — Local Embeddings + FAISS")

# Check OpenAI key (for LLM generation)
OPENAI_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
elif os.environ.get("OPENAI_API_KEY"):
    OPENAI_KEY = os.environ["OPENAI_API_KEY"]

if OPENAI_KEY:
    openai.api_key = OPENAI_KEY
    st.sidebar.success("🔑 OpenAI key loaded (for generation).")
else:
    st.sidebar.warning("⚠️ No OpenAI key found. Add OPENAI_API_KEY to Streamlit Secrets or set locally. (You can still build the FAISS index; generation will be disabled.)")

# Sidebar inputs
st.sidebar.header("Ingest options")
ingest_mode = st.sidebar.radio("Ingest method", ["Upload PDFs", "Excel Links (download)"])
chunk_size = st.sidebar.slider("Chunk size (chars)", 500, 2000, 1000)
chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", 50, 500, 200)
top_k = st.sidebar.slider("Retriever top_k", 1, 10, 5)
sentence_model = os.environ.get("SENTENCE_MODEL", "all-MiniLM-L6-v2")

# Upload PDFs
if ingest_mode == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload one or more PDF brochures", type=["pdf"], accept_multiple_files=True)
    if st.button("Index uploaded files"):
        if not uploaded_files:
            st.warning("Please upload PDFs first.")
        else:
            all_chunks = []
            for f in uploaded_files:
                # Save temporarily then process
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                chunks = pdf_to_chunks(tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_chunks.extend(chunks)
            if not all_chunks:
                st.error("No text extracted from uploaded PDFs. If PDFs are scanned images, OCR is required.")
            else:
                with st.spinner("Indexing documents (this may take a while)..."):
                    create_or_get_faiss(all_chunks, model_name=sentence_model)
                st.success(f"Indexed {len(all_chunks)} chunks. FAISS index saved to disk.")

# Excel download flow
if ingest_mode == "Excel Links (download)":
    excel = st.file_uploader("Upload Excel with brochure links (columns: PSM_ID, Brochure_Link)", type=["xlsx", "xls"])
    if excel:
        uploads_dir = Path("uploads"); uploads_dir.mkdir(exist_ok=True)
        ex_file = uploads_dir / excel.name
        with open(ex_file, "wb") as out:
            out.write(excel.getbuffer())
        limit = st.number_input("Limit rows to download (0 = all)", min_value=0, value=100)
        if st.button("Download & Index"):
            st.info("Starting download & indexing. This may take a long time for many rows.")
            report = download_from_excel(str(ex_file), limit=limit if limit > 0 else None)
            report.to_csv("download_report.csv", index=False)
            st.write("Download report saved to download_report.csv")
            downloads = report[report["status"].isin(["pdf", "image->pdf", "raw"])]["file"].tolist()
            all_chunks = []
            for p in downloads:
                try:
                    chunks = pdf_to_chunks(p, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    all_chunks.extend(chunks)
                except Exception as e:
                    st.write(f"Failed to process {p}: {e}")
            if all_chunks:
                with st.spinner("Indexing downloaded documents..."):
                    create_or_get_faiss(all_chunks, model_name=sentence_model)
                st.success(f"Indexed {len(all_chunks)} chunks from downloaded files.")

st.markdown("---")
st.header("💬 Ask questions (RAG)")

# Check if FAISS index exists
from pathlib import Path
faiss_index_exists = Path(os.environ.get("FAISS_INDEX_PATH", "./faiss_index.index")).exists() and Path(os.environ.get("FAISS_DOCS_PATH", "./faiss_docs.pkl")).exists()

if not faiss_index_exists:
    st.info("No FAISS index found. Upload / download brochures and click Index to create one.")
else:
    query = st.text_input("Ask a question about the uploaded brochures")
    if st.button("Ask") and query:
        # Run retrieval
        hits = search(query, top_k=top_k, model_name=sentence_model)
        if not hits:
            st.info("No relevant content found.")
        else:
            # Build context from retrieved hits
            context_parts = []
            for i, h in enumerate(hits, start=1):
                src = h["metadata"].get("source_file", "unknown")
                page = h["metadata"].get("page", None)
                header = f"Source: {src}" + (f" (page {page})" if page else "")
                context_parts.append(f"{header}\n{h['text']}\n")
            context = "\n\n".join(context_parts)

            # Compose prompt
            prompt = f"""You are a helpful property brochure assistant. Use the provided context (below) to answer the user's question. If the answer is not in the context, say you don't know. Do NOT hallucinate.

Context:
{context}

Question: {query}

Answer:"""

            if not OPENAI_KEY:
                st.error("OpenAI API key is not set — cannot generate an answer. Add OPENAI_API_KEY to Streamlit secrets.")
            else:
                with st.spinner("Generating answer from LLM..."):
                    try:
                        # Use chat completion (gpt-3.5-turbo). Adjust model as desired.
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=512,
                            temperature=0.0,
                        )
                        answer = response["choices"][0]["message"]["content"].strip()
                        st.subheader("Answer")
                        st.write(answer)

                        st.subheader("Sources (top results)")
                        for i, h in enumerate(hits, start=1):
                            src = h["metadata"].get("source_file", "unknown")
                            page = h["metadata"].get("page", "")
                            st.markdown(f"**{i}. {src} (page {page})** — score: {h['score']:.4f}")
                            st.caption(h["text"][:400] + ("..." if len(h["text"]) > 400 else ""))
                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")
