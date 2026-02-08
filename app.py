import os
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import streamlit as st
from typing import List
import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Import Google Gemini integration
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception as e:
    st.error("⚠️ Could not import ChatGoogleGenerativeAI. Run: pip install langchain-google-genai")
    raise e

# -------------------------
# Helper functions
# -------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_pages = []
    for page in doc:
        page_text = page.get_text() or ""
        text_pages.append(page_text)
    doc.close()
    return "\n\n".join(text_pages)


def build_documents_from_uploaded_files(uploaded_files) -> List[Document]:
    """Convert uploaded PDFs to LangChain Documents."""
    docs = []
    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        raw_bytes = uploaded_file.read()
        text = extract_text_from_pdf_bytes(raw_bytes)
        if not text.strip():
            st.warning(f"No extractable text in {name}. (Scanned PDFs may require OCR.)")
            continue
        docs.append(Document(page_content=text, metadata={"source": name}))
    return docs


def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            md = dict(d.metadata)
            md.update({"chunk": i})
            split_docs.append(Document(page_content=c, metadata=md))
    return split_docs


# -------------------------
# Streamlit UI
# -------------------------
# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="RAG Research Assistant", layout="wide", page_icon="📚")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    /* Global Styles */
    h1 {
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #2563EB; /* Primary Blue */
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        padding: 0.5rem;
    }
    /* Custom Card Style for Sources */
    .source-card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Clean Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=80)
with col2:
    st.title("Research Assistant AI")
    st.markdown("**Powered by Google Gemini 2.5** • *Upload your PDFs and ask anything.*")


with st.sidebar:
    st.header("⚙️ Settings")
    # API Key Handling
    # If the key is already in the environment (e.g. from .env), don't ask for it again.
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if GOOGLE_API_KEY:
        st.success("✅ API Key loaded from .env file")
    else:
        GOOGLE_API_KEY = st.text_input("🔑 Google API Key", type="password")
        if not GOOGLE_API_KEY:
             st.warning("Please enter your API Key")
             st.stop()
    
    model_name = st.selectbox(
        "🧠 Google Model Name", 
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest"],
        index=0
    )
    k_results = st.number_input("Retriever top-k results", min_value=1, max_value=10, value=5)
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
    st.markdown("---")
    st.markdown("⚠️ If PDFs are scanned images, run OCR before uploading.")
    st.caption("Get your free API key here: https://aistudio.google.com/app/apikey")

# The key is already handled in the sidebar logic above


# -------------------------
# Step 1 — Upload PDFs
# -------------------------
st.subheader("Step 1 — Upload PDF(s)")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) uploaded. Click 'Build Index' to start processing.")
    if st.button("Build Index"):
        with st.spinner("🔍 Extracting text from PDFs..."):
            docs = build_documents_from_uploaded_files(uploaded_files)

        if not docs:
            st.error("No text extracted. Try another file or run OCR first.")
        else:
            split_docs = chunk_documents(docs, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
            st.success(f"✅ Extracted {len(split_docs)} chunks of text.")

            # Build embeddings and store in Chroma
            st.info("🔢 Generating embeddings and building Chroma index...")
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            texts = [d.page_content for d in split_docs]
            metadatas = [d.metadata for d in split_docs]

            try:
                vectordb = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
                st.session_state["vectordb"] = vectordb
                st.success("✅ Vector index built! Proceed to Step 2.")
            except Exception as e:
                st.error(f"Error creating Chroma store: {e}")
else:
    st.info("Upload your PDFs above to begin.")

# -------------------------
# Step 2 — Ask Questions
# -------------------------
st.markdown("---")
st.subheader("💬 Ask Your Questions")


if "vectordb" not in st.session_state:
    st.info("Build an index first in Step 1.")
else:
    query = st.text_input("Ask a question about your uploaded PDFs:")
    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            vectordb: Chroma = st.session_state["vectordb"]
            
            # 1. Retrieve relevant documents (Simpler: direct similarity search)
            with st.spinner("🔍 Searching documents..."):
                docs = vectordb.similarity_search(query, k=int(k_results))
                
            if not docs:
                st.warning("No relevant documents found. Try rephrasing your question.")
            else:
                # 2. Build the context string
                context_text = "\n\n".join([f"Source ({doc.metadata.get('source', 'unknown')}): {doc.page_content}" for doc in docs])
                
                # 3. Create the prompt (Simple string formatting)
                system_prompt = "You are a helpful research assistant. Use the provided context to answer the user's question. If the answer is not in the context, say so."
                user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
                
                final_prompt = [
                    ("system", system_prompt),
                    ("human", user_prompt)
                ]

                # 4. Call the LLM
                os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0.0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )
                    
                    with st.spinner("🧠 Thinking..."):
                        response = llm.invoke(final_prompt)
                        answer = response.content

                    # Display Answer
                    st.markdown("### 🧩 Answer")
                    st.success(answer)


                    # Display Sources (for verification)
                    st.markdown("### 📚 Sources Used")
                    for i, doc in enumerate(docs):
                        with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'unknown')}"):
                            st.write(doc.page_content)
                    
                    # Explain how it works to the interviewer
                    with st.expander("🛠️ How this works (For Interview Explanation)"):
                        st.markdown(f"""
                        **1. Semantic Search:** We searched the vector database for chunks similar to: *"{query}"*
                        **2. Context Construction:** We found **{len(docs)}** relevant chunks and combined them into a context string.
                        **3. LLM Prompting:** We sent the following prompt to the **{model_name}** model:
                        
                        ```text
                        System: {system_prompt}
                        
                        User: Context:
                        [... {len(context_text)} chars of text ...]
                        
                        Question: {query}
                        ```
                        """)

                except Exception as e:
                    st.error(f"Error running Google Gemini: {e}")
