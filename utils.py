import os
import base64
import io
import tempfile

import requests
import streamlit as st
import pandas as pd
import fitz
from PIL import Image
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from models import StreamHandler

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

VISION_KEYWORDS = ["vision", "llava", "minicpm", "moondream", "gemma3"]
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Ollama ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def get_ollama_models() -> list:
    """Fetch available Ollama models. Cached for 30 seconds to avoid repeated HTTP calls."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
        return []
    except Exception:
        return []


def has_vision_model(models: list) -> bool:
    return any(any(k in m for k in VISION_KEYWORDS) for m in models)


# ── PDF Loading ──────────────────────────────────────────────────────────────

def smart_load_pdf(file_path: str, file_name: str) -> tuple:
    """
    Load a PDF with automatic OCR fallback for scanned pages.

    Pages with fewer than 50 characters are treated as scanned
    and processed through pytesseract OCR at 300 DPI.

    Returns:
        tuple: (list of Documents, scanned_page_count)
    """
    docs = []
    scanned = 0
    try:
        loader = PyPDFium2Loader(file_path)
        for doc in loader.load():
            if len(doc.page_content.strip()) < 50 and OCR_AVAILABLE:
                scanned += 1
                page_num = doc.metadata.get("page", 0)
                pdf_doc = fitz.open(file_path)
                pix = pdf_doc[page_num].get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img).strip()
                if ocr_text:
                    doc.page_content = ocr_text
                    doc.metadata["ocr"] = True
                pdf_doc.close()
            docs.append(doc)
    except Exception as e:
        st.warning(f"Error loading {file_name}: {e}")
    return docs, scanned


# ── Semantic Chunking ────────────────────────────────────────────────────────

def semantic_chunk(docs: list) -> list:
    """
    Split documents into chunks at semantic topic boundaries.

    Uses SemanticChunker from langchain_experimental when available,
    which breaks chunks where the topic actually changes rather than
    at fixed token counts. Falls back to RecursiveCharacterTextSplitter.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )
        chunks = chunker.split_documents(docs)
        return chunks if chunks else _fallback_chunk(docs)
    except Exception:
        return _fallback_chunk(docs)


def _fallback_chunk(docs: list) -> list:
    """Fallback chunking using fixed token size with overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    return splitter.split_documents(docs)


# ── Hybrid Retrieval ─────────────────────────────────────────────────────────

def hybrid_retrieve(question: str, vector_db, all_chunks: list, k: int = 6) -> list:
    """
    Combine FAISS semantic search with BM25 keyword search.

    FAISS handles conceptual similarity while BM25 catches exact
    keyword matches that semantic search might miss. Results from
    both are merged and deduplicated.

    Args:
        question:   User query string
        vector_db:  FAISS vector store instance
        all_chunks: All document chunks for BM25 indexing
        k:          Number of results to return

    Returns:
        list: Merged unique documents from both retrievers
    """
    faiss_docs = vector_db.similarity_search(question, k=k)

    bm25_docs = []
    if BM25_AVAILABLE and all_chunks:
        try:
            tokenized = [doc.page_content.lower().split() for doc in all_chunks]
            bm25      = BM25Okapi(tokenized)
            scores    = bm25.get_scores(question.lower().split())
            top_idx   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            bm25_docs = [all_chunks[i] for i in top_idx]
        except Exception:
            pass

    seen, merged = set(), []
    for doc in faiss_docs + bm25_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged[:k]


# ── Reranker ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_reranker():
    """Load the CrossEncoder reranker model. Cached for the session lifetime."""
    if not RERANKER_AVAILABLE:
        return None
    try:
        return CrossEncoder(RERANKER_MODEL)
    except Exception:
        return None


def rerank_docs(question: str, docs: list, top_k: int = 3) -> list:
    """
    Re-order retrieved documents by true relevance using a CrossEncoder.

    Unlike bi-encoder models (FAISS), a CrossEncoder reads the question
    and each document together, producing a more accurate relevance score.

    Args:
        question: User query string
        docs:     Candidate documents from retrieval
        top_k:    Number of top documents to return

    Returns:
        list: Top-k documents sorted by CrossEncoder relevance score
    """
    if not docs:
        return docs

    reranker = load_reranker()
    if reranker is None:
        return docs[:top_k]

    try:
        pairs  = [[question, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    except Exception:
        return docs[:top_k]


# ── Table Extraction ─────────────────────────────────────────────────────────

@st.cache_data
def extract_tables(file_bytes: bytes, file_name: str) -> list:
    """
    Extract tables from a PDF file.

    Accepts file_bytes instead of a file object so Streamlit can
    use the content hash as a cache key. Both the Tables tab and
    Charts tab share this cache — no double processing.
    """
    import pdfplumber
    tables = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        with pdfplumber.open(tmp_path) as pdf:
            for pn, page in enumerate(pdf.pages, 1):
                for tbl in page.extract_tables():
                    if tbl and len(tbl) > 1:
                        try:
                            headers = [str(h) if h else f"Col{i}" for i, h in enumerate(tbl[0])]
                            tables.append({"page": pn, "df": pd.DataFrame(tbl[1:], columns=headers)})
                        except Exception:
                            pass
        os.unlink(tmp_path)
    except Exception as e:
        st.warning(f"Table extraction error: {e}")
    return tables


# ── Image Extraction ─────────────────────────────────────────────────────────

def extract_images(file_bytes: bytes) -> list:
    """Extract all images larger than 50x50px from a PDF."""
    images = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        for pn in range(len(doc)):
            for ii, img in enumerate(doc[pn].get_images(full=True)):
                bi  = doc.extract_image(img[0])
                pil = Image.open(io.BytesIO(bi["image"]))
                if pil.width > 50 and pil.height > 50:
                    images.append({
                        "page": pn + 1, "image": pil,
                        "index": ii + 1, "size": f"{pil.width}x{pil.height}"
                    })
        doc.close()
        os.unlink(tmp_path)
    except Exception as e:
        st.warning(f"Image extraction error: {e}")
    return images


# ── Vision Analysis ──────────────────────────────────────────────────────────

def analyze_image(image, available_models: list, question: str) -> str:
    """Send an image to the Ollama vision model for analysis."""
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        vm    = [m for m in available_models if any(k in m for k in VISION_KEYWORDS)]
        model = vm[0] if vm else available_models[0]
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": question, "images": [img_b64], "stream": False},
            timeout=120,
        )
        return r.json().get("response", "Could not analyze.") if r.status_code == 200 else "Failed."
    except Exception as e:
        return f"Error: {e}"


# ── Chat ─────────────────────────────────────────────────────────────────────

def chat_with_docs(question: str, context_docs: list, messages: list,
                   model_name: str, placeholder) -> str:
    """
    Generate a streaming response from the LLM using retrieved context.

    Uses a rolling window of the last 10 messages to prevent context
    overflow while maintaining conversational continuity.
    """
    context = "\n\n".join([
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in context_docs
    ])

    history_str = ""
    if messages:
        recent      = messages[-10:]
        history_str = "\n".join([
            f"User: {m['content']}" if m["role"] == "user"
            else f"Assistant: {m['content']}"
            for m in recent
        ])

    full_prompt = f"""You are a highly intelligent AI assistant analyzing PDF documents.

Rules:
- Give a helpful, direct answer
- Do NOT say "based on the context" or "according to the document"
- Always respond in the same language as the question
- Mention the source page when relevant (e.g. "On page 5...")
- If unsure, say so honestly

Document Context:
{context}

Conversation History:
{history_str}

Question: {question}

Answer:"""

    handler = StreamHandler(placeholder)
    llm = ChatOllama(
        model=model_name,
        temperature=0.3,
        num_predict=512,
        streaming=True,
        callbacks=[handler],
    )
    llm.invoke(full_prompt)
    return handler.text
