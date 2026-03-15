"""
Private AI v4.0 — Enterprise Multi-PDF Chatbot
Run: streamlit run main.py
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from config import set_page_config, inject_css
from utils import get_ollama_models, VISION_KEYWORDS
from sidebar import render_sidebar
from tabs import render_tabs

# Page setup
set_page_config()
inject_css()

# Session state defaults
for key, val in {
    "messages":       [],
    "vector_db":      None,
    "page_tree":      None,
    "all_chunks":     [],
    "rag_mode":       "faiss",
    "kb_initialized": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Load embedding model once per session
if "embeddings" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

# Sidebar
selected_model, uploaded_files = render_sidebar()
available_models = get_ollama_models()

# Hero header
st.markdown('<div class="hero-title">Private AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">ENTERPRISE MULTI-PDF INTELLIGENCE · v4.0 FULL RAG PIPELINE</div>',
    unsafe_allow_html=True,
)
badge_html = "".join([
    f'<span class="hero-badge {cls}">{label}</span>'
    for label, cls in [
        ("⚡ FAISS Vector DB", ""),
        ("🌲 PageIndex RAG",   "pi-badge"),
        ("🧠 Auto Smart RAG",  "pi-badge"),
        ("🔀 Hybrid Search",   ""),
        ("🎯 Reranker",        ""),
        ("🔒 100% Private",    ""),
    ]
])
st.markdown(badge_html, unsafe_allow_html=True)

# Main content
if uploaded_files:
    render_tabs(uploaded_files, selected_model, available_models)
else:
    features = [
        ("🧠", "Auto Smart RAG",  "Automatically selects FAISS or PageIndex based on document size"),
        ("🌲", "PageIndex RAG",   "LLM reasons through a document tree — no vector math"),
        ("🔀", "Hybrid Search",   "Combines FAISS semantic search with BM25 keyword matching"),
        ("🎯", "Reranker",        "CrossEncoder re-orders results by true relevance"),
        ("💬", "Smart Chat",      "Ask questions across 5000+ PDFs with conversation memory"),
        ("🔍", "OCR Engine",      "Automatically reads scanned and image-based PDF pages"),
        ("👁️", "Vision AI",       "Analyze charts and diagrams with multimodal AI"),
        ("🔒", "Zero Data Leaks", "100% local — documents never leave your machine"),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Vector DB", "FAISS ⚡", "Large docs")
    with c2:
        st.metric("Smart RAG", "PageIndex 🌲", "Small docs")
    with c3:
        st.metric("Auto Switch", "≤50 pages 🧠", "No manual choice")
    vm = [m for m in available_models if any(k in m for k in VISION_KEYWORDS)]
    with c4:
        st.metric(
            "Vision AI",
            vm[0].split(":")[0] if vm else "Not installed",
            "Charts & Images" if vm else "ollama pull llama3.2-vision"
        )
