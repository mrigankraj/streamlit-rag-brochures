import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
from processor import load_pdf, split_documents
from vectorstore import create_or_get_chroma, get_retriever
from downloader import download_from_excel
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title='Brochure RAG — Streamlit', layout='wide')
st.title('📄 Brochure RAG — Upload or Excel → Index → Ask')

with st.sidebar:
    st.header('Ingest options')
    source = st.radio('Choose ingestion method:', ['Upload PDFs', 'Excel Links (download)'])
    chunk_size = st.slider('Chunk size (chars)', 500, 2000, 1000)
    overlap = st.slider('Chunk overlap', 50, 500, 200)

if source == 'Upload PDFs':
    uploaded_files = st.file_uploader('Upload PDFs', type=['pdf'], accept_multiple_files=True)
    if st.button('Index uploaded files'):
        if not uploaded_files:
            st.warning('Please upload files first.')
        else:
            docs = []
            for f in uploaded_files:
                path = Path('uploads'); path.mkdir(exist_ok=True)
                dest = path / f.name
                with open(dest, 'wb') as out: out.write(f.getbuffer())
                loaded = load_pdf(str(dest))
                splits = split_documents(loaded, chunk_size=chunk_size, chunk_overlap=overlap)
                for s in splits: s.metadata['source_file'] = dest.name
                docs.extend(splits)
            if docs:
                create_or_get_chroma(docs, persist=True)
                st.success(f'✅ Indexed {len(docs)} chunks from {len(uploaded_files)} files.')

if source == 'Excel Links (download)':
    excel = st.file_uploader('Upload Excel with links', type=['xlsx','xls'])
    if excel:
        ex_path = Path('uploads'); ex_path.mkdir(exist_ok=True)
        ex_file = ex_path / excel.name
        with open(ex_file, 'wb') as out: out.write(excel.getbuffer())
        limit = st.number_input('Limit downloads (0 = all)', min_value=0, value=100)
        if st.button('Download & Index'):
            report = download_from_excel(str(ex_file), limit=limit if limit>0 else None)
            report.to_csv('download_report.csv', index=False)
            st.write('📊 Report saved → download_report.csv')
            downloads = report[report['status'].isin(['pdf','image->pdf','raw'])]['file'].tolist()
            docs = []
            for p in downloads:
                loaded = load_pdf(p)
                splits = split_documents(loaded, chunk_size=chunk_size, chunk_overlap=overlap)
                for s in splits: s.metadata['source_file'] = Path(p).name
                docs.extend(splits)
            if docs:
                create_or_get_chroma(docs, persist=True)
                st.success(f'✅ Indexed {len(docs)} chunks from {len(downloads)} files.')

st.markdown('---')
st.header('💬 Ask questions')
question = st.text_input('Enter your question about the brochures')
if st.button('Ask'):
    if not question:
        st.warning('Type a question first!')
    else:
        retriever = get_retriever(top_k=5)
        llm = ChatOpenAI(temperature=0.0)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
        with st.spinner('Thinking...'):
            answer = qa.run(question)
        st.subheader('Answer:')
        st.write(answer)
