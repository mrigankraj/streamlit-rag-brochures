# Streamlit RAG — Local Embeddings (sentence-transformers) + FAISS

This app:
- Ingests PDF brochures (upload or Excel links).
- Extracts text and chunks it.
- Creates local embeddings using sentence-transformers.
- Builds a FAISS index (persisted).
- Allows natural-language queries and uses OpenAI (optional) to generate final answers.

## Run locally
1. python -m venv .venv
2. source .venv/bin/activate
3. pip install -r requirements.txt
4. copy .env.example -> .env and add OPENAI_API_KEY (if using generation)
5. streamlit run app.py
