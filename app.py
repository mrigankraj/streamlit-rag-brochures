import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from vectorstore import create_or_get_faiss, get_retriever

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Brochure QA", layout="wide")

st.title("📄 Brochure Question Answering App (FAISS + OpenAI)")

# Sidebar for file upload
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF brochures", type=["pdf"], accept_multiple_files=True
)

# Process uploaded documents
if uploaded_files:
    from langchain_community.document_loaders import PyPDFLoader

    docs = []
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        docs.extend(loader.load())

    st.sidebar.success(f"✅ Loaded {len(docs)} pages from {len(uploaded_files)} files")

    # Build / update FAISS index
    with st.spinner("🔎 Indexing documents..."):
        create_or_get_faiss(docs)
    st.sidebar.success("📚 Documents indexed in FAISS!")

# Load retriever
retriever = get_retriever(top_k=5)

# Build QA chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_template = """Use the following context to answer the question at the end. 
If you don’t know the answer, just say you don’t know — don’t make anything up.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)

# Chat interface
st.header("💬 Ask a Question")
user_query = st.text_input("Type your question about the brochures...")

if user_query:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain.invoke({"query": user_query})

        st.subheader("Answer")
        st.write(result["result"])

        # Show sources
        if "source_documents" in result:
            st.subheader("Sources")
            for i, doc in enumerate(result["source_documents"], start=1):
                st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                st.caption(doc.page_content[:300] + "...")
