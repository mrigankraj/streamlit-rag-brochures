# streamlit-rag-brochures

# 📄 Brochure RAG — Streamlit App

This app lets you:
- Upload PDFs **or** provide an Excel file with brochure links  
- Automatically download and convert to PDFs if needed  
- Chunk + embed with OpenAI  
- Store in Chroma vector DB  
- Query with RAG pipeline  

## 🚀 Run Locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your keys
streamlit run app.py
