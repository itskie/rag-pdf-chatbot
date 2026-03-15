import os
import tempfile

import streamlit as st
from langchain_community.vectorstores import FAISS

from models import PageIndexTree
from utils import (
    get_ollama_models,
    has_vision_model,
    smart_load_pdf,
    semantic_chunk,
    BM25_AVAILABLE,
    RERANKER_AVAILABLE,
    OCR_AVAILABLE,
    VISION_KEYWORDS,
)

# Pages at or below this threshold use PageIndex; above it use FAISS
PAGEINDEX_THRESHOLD = 50


def detect_rag_mode(total_pages: int) -> str:
    """
    Auto-select retrieval mode based on document size.

    Small documents (<=50 pages) benefit from PageIndex reasoning.
    Large documents (>50 pages) use FAISS for speed and scalability.
    """
    return "pageindex" if total_pages <= PAGEINDEX_THRESHOLD else "faiss"


def render_sidebar():
    """
    Render the sidebar UI.

    Returns:
        tuple: (selected_model, uploaded_files)
    """
    available_models = get_ollama_models()

    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🛡️ Private AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-version">v4.4 — Full RAG Pipeline</div>', unsafe_allow_html=True)

        # Model selector
        if available_models:
            model_labels = [
                f"{m} 👁️" if any(k in m for k in VISION_KEYWORDS) else m
                for m in available_models
            ]
            default_idx = next(
                (i for i, m in enumerate(available_models)
                 if not any(k in m for k in VISION_KEYWORDS)), 0
            )
            selected_label = st.selectbox("🤖 Active Model", options=model_labels, index=default_idx)
            selected_model = selected_label.replace(" 👁️", "")
        else:
            st.error("⚠️ Ollama not running! Run: `ollama serve`")
            selected_model = "No model"

        # Auto Smart RAG status
        st.markdown("---")
        st.markdown("**🧠 Auto Smart RAG**")
        if st.session_state.kb_initialized:
            mode = st.session_state.rag_mode
            if mode == "pageindex":
                st.markdown('<span class="status-ok">🌲 PageIndex Active</span>', unsafe_allow_html=True)
                st.caption(f"<={PAGEINDEX_THRESHOLD} pages — reasoning mode.")
            else:
                st.markdown('<span class="status-ok">⚡ FAISS Active</span>', unsafe_allow_html=True)
                st.caption(f">{PAGEINDEX_THRESHOLD} pages — speed mode.")
        else:
            st.caption(
                f"<={PAGEINDEX_THRESHOLD} pages → 🌲 PageIndex\n\n"
                f">{PAGEINDEX_THRESHOLD} pages → ⚡ FAISS"
            )

        # Status badges
        st.markdown("---")
        if available_models:
            mode_label = "☁️ Cloud" if ":cloud" in selected_model else "🔒 Local"
            st.markdown(f'<span class="model-pill">{mode_label} · {selected_model}</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        if has_vision_model(available_models):
            st.markdown('<span class="status-ok">👁️ Vision Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">👁️ No Vision Model</span>', unsafe_allow_html=True)

        if OCR_AVAILABLE:
            st.markdown('<span class="status-ok">🔍 OCR Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">🔍 OCR Not Available</span>', unsafe_allow_html=True)

        if BM25_AVAILABLE:
            st.markdown('<span class="status-ok">🔀 Hybrid Search Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">🔀 BM25 Not Installed</span>', unsafe_allow_html=True)

        if RERANKER_AVAILABLE:
            st.markdown('<span class="status-ok">🎯 Reranker Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">🎯 Reranker Not Installed</span>', unsafe_allow_html=True)

        if st.session_state.kb_initialized:
            kb_label = "🌲 PageIndex" if st.session_state.rag_mode == "pageindex" else "⚡ FAISS"
            st.markdown(f'<span class="status-ok">{kb_label} KB Ready</span>', unsafe_allow_html=True)

        st.markdown("---")

        uploaded_files = st.file_uploader(
            "📄 Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.caption(f"📚 {len(uploaded_files)} file(s) selected")

        if st.button("⚡ Initialize Knowledge Base", use_container_width=True):
            _initialize_kb(uploaded_files, selected_model, available_models)

        st.markdown("---")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="feature-desc" style="color:#334155">Built by Shobhit Singh<br>'
            'github.com/itskie</div>',
            unsafe_allow_html=True,
        )

    return selected_model, uploaded_files


def _initialize_kb(uploaded_files, selected_model, available_models):
    """
    Build the knowledge base from uploaded PDFs.

    Automatically selects FAISS or PageIndex based on document size.
    Applies semantic chunking before building the FAISS index.
    """
    if not available_models:
        st.error("Start Ollama first!")
        return
    if not uploaded_files:
        st.error("Upload at least one PDF!")
        return

    all_docs      = []
    total_scanned = 0
    progress      = st.progress(0, text="Loading PDFs...")

    for i, file in enumerate(uploaded_files):
        progress.progress(
            i / len(uploaded_files),
            text=f"Loading {file.name} ({i+1}/{len(uploaded_files)})"
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        try:
            docs, scanned = smart_load_pdf(tmp_path, file.name)
            all_docs.extend(docs)
            total_scanned += scanned
        finally:
            os.remove(tmp_path)

    total_pages = len(all_docs)
    auto_mode   = detect_rag_mode(total_pages)
    st.session_state.rag_mode = auto_mode
    ocr_msg = f" · {total_scanned} OCR pages" if total_scanned else ""

    if auto_mode == "faiss":
        progress.progress(0.6, text="Applying semantic chunking...")
        chunks = semantic_chunk(all_docs)

        progress.progress(0.8, text=f"Building FAISS index — {len(chunks)} chunks...")
        from langchain_huggingface import HuggingFaceEmbeddings
        vector_db = FAISS.from_documents(chunks, st.session_state.embeddings)

        st.session_state.vector_db  = vector_db
        st.session_state.all_chunks = chunks
        st.session_state.page_tree  = None
        progress.progress(1.0, text="✅ FAISS Ready!")
        st.success(
            f"✅ {len(uploaded_files)} PDF · {total_pages} pages · "
            f"{len(chunks)} chunks{ocr_msg} · ⚡ FAISS auto-selected"
        )

    else:
        progress.progress(0.4, text=f"Building PageIndex tree — {total_pages} pages...")
        tree = PageIndexTree(all_docs, model_name=selected_model, batch_size=5)

        def update_progress(p):
            progress.progress(
                0.4 + p * 0.6,
                text=f"Indexing sections... {int(p * 100)}%"
            )

        tree.build(progress_callback=update_progress)
        st.session_state.page_tree  = tree
        st.session_state.vector_db  = None
        st.session_state.all_chunks = []
        progress.progress(1.0, text="✅ PageIndex Ready!")
        st.success(
            f"✅ {len(uploaded_files)} PDF · {total_pages} pages · "
            f"{len(tree.tree)} sections{ocr_msg} · 🌲 PageIndex auto-selected"
        )

    st.session_state.kb_initialized = True
    st.session_state.messages       = []
