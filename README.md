## RAG Research Assistant (PDF Q&A) — Streamlit + Chroma + Groq

Upload PDF research papers, build a vector index, and ask questions about the content using a Retrieval-Augmented Generation (RAG) pipeline.

### Tech stack
- **UI**: Streamlit
- **PDF text extraction**: PyMuPDF (`pymupdf` / `fitz`)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: ChromaDB
- **LLM**: Groq via `langchain-groq`

### Run locally
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your Groq API key (PowerShell):

```powershell
$env:GROQ_API_KEY="YOUR_KEY_HERE"
```

3. Start the app:

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

### Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub
2. Go to Streamlit Community Cloud and create a new app from this repo (`app.py`)
3. Add your key in **Secrets**:

```toml
GROQ_API_KEY="YOUR_KEY_HERE"
```

### Notes
- If your PDFs are scanned images, you’ll need OCR first (otherwise no text will be extracted).
- If Groq returns a “model decommissioned” error, choose a supported model in the app sidebar.

