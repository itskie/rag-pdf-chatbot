import streamlit as st
import os
import requests
import pandas as pd
import plotly.express as px
import pdfplumber
import fitz
from PIL import Image
import io
import tempfile
import time
import base64

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ─── Page Config ───
st.set_page_config(page_title="Private AI v3.0", page_icon="🛡️", layout="wide")

# ─── Premium Dark CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    background-color: #080C14 !important;
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer {visibility: hidden;}
.stDeployButton {display: none;}


/* ── Main container ── */
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 0%, #60A5FA 50%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #64748B;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

.hero-badge {
    display: inline-block;
    background: rgba(96, 165, 250, 0.1);
    border: 1px solid rgba(96, 165, 250, 0.2);
    color: #60A5FA;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-right: 0.5rem;
    margin-bottom: 1.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E293B !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}

/* ── Sidebar header ── */
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    color: #F8FAFC;
    margin-bottom: 0.2rem;
}

.sidebar-version {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    margin-bottom: 1.5rem;
}

/* ── Status badges ── */
.status-ok {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.2);
    color: #4ADE80;
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.4rem;
    display: block;
}

.status-warn {
    background: rgba(234, 179, 8, 0.1);
    border: 1px solid rgba(234, 179, 8, 0.2);
    color: #FDE047;
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.4rem;
    display: block;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0D1117 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0D1117 !important;
    border: 1px dashed #1E293B !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1E40AF, #7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.3) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #0D1117 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    margin-bottom: 0.8rem !important;
    padding: 1rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0D1117 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] textarea {
    background: #0D1117 !important;
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #0D1117 !important;
    border-bottom: 1px solid #1E293B !important;
    gap: 0.5rem !important;
}

[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #64748B !important;
    border: 1px solid transparent !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(30,64,175,0.2), rgba(124,58,237,0.2)) !important;
    color: #60A5FA !important;
    border-color: #1E293B !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0D1117 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}

[data-testid="stMetricValue"] {
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] {
    color: #4ADE80 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0D1117 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
}

/* ── Info/Success/Warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1E40AF, #7C3AED) !important;
}

/* ── Divider ── */
hr {
    border-color: #1E293B !important;
}

/* ── Feature card ── */
.feature-card {
    background: #0D1117;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}

.feature-card:hover {
    border-color: #334155;
}

.feature-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.feature-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    color: #F1F5F9;
    margin-bottom: 0.3rem;
}

.feature-desc {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #64748B;
    line-height: 1.5;
}

/* ── Model pill ── */
.model-pill {
    display: inline-block;
    background: rgba(96, 165, 250, 0.08);
    border: 1px solid rgba(96, 165, 250, 0.15);
    color: #93C5FD;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)


# ─── Auto Detect Ollama Models ───
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        return []
    except Exception:
        return []

available_models = get_ollama_models()

# ─── Vision check ───
def has_vision_model():
    vision_keywords = ["vision", "llava", "minicpm", "moondream", "gemma3"]
    return any(any(k in m for k in vision_keywords) for m in available_models)

# ─── Custom Prompt ───
custom_prompt = PromptTemplate(
    template="""You are a highly intelligent and friendly AI assistant analyzing PDF documents.
Answer the user's question clearly and conversationally. Be concise but complete.

Rules:
- Give a helpful, direct answer
- Use simple language  
- Do NOT say "based on the context" or "according to the document"
- Answer naturally and confidently
- If you don't know something, say so honestly
- VERY IMPORTANT: Always respond in the SAME language as the user's question (Hindi, English, or Hinglish)
- Always mention the source page number when possible

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:""",
    input_variables=["context", "chat_history", "question"]
)

# ─── Session State ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
if "chain" not in st.session_state:
    st.session_state.chain = None
if "embeddings" not in st.session_state:
    with st.spinner("Loading embeddings..."):
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── Vision: Analyze image ───
def analyze_image(image, question="Describe this image in detail. If it's a chart or graph, explain what data it shows, trends, and key insights."):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        vision_models = [m for m in available_models if any(k in m for k in ["vision", "llava", "minicpm", "moondream", "gemma3"])]
        model = vision_models[0] if vision_models else available_models[0]

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": question, "images": [img_b64], "stream": False},
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "Could not analyze.")
        return "Analysis failed."
    except Exception as e:
        return f"Error: {e}"

# ─── Smart PDF Loader ───
def smart_load_pdf(file_path, file_name):
    docs = []
    scanned_pages = 0
    try:
        loader = PyPDFium2Loader(file_path)
        loaded = loader.load()
        for doc in loaded:
            if len(doc.page_content.strip()) < 50 and OCR_AVAILABLE:
                scanned_pages += 1
                page_num = doc.metadata.get("page", 0)
                pdf_doc = fitz.open(file_path)
                page = pdf_doc[page_num]
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img).strip()
                if ocr_text:
                    doc.page_content = ocr_text
                    doc.metadata["ocr"] = True
                pdf_doc.close()
            docs.append(doc)
    except Exception as e:
        st.warning(f"⚠️ Error loading {file_name}: {e}")
    return docs, scanned_pages

# ─── Extract Tables ───
def extract_tables(file):
    tables = []
    try:
        file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        with pdfplumber.open(tmp_path) as pdf:
            for pn, page in enumerate(pdf.pages, 1):
                for tbl in page.extract_tables():
                    if tbl and len(tbl) > 1:
                        try:
                            headers = [str(h) if h else f"Col{i}" for i, h in enumerate(tbl[0])]
                            tables.append({"page": pn, "df": pd.DataFrame(tbl[1:], columns=headers)})
                        except: pass
        os.unlink(tmp_path)
    except Exception as e:
        st.warning(f"Table error: {e}")
    return tables

# ─── Extract Images ───
def extract_images(file):
    images = []
    try:
        file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        for pn in range(len(doc)):
            for ii, img in enumerate(doc[pn].get_images(full=True)):
                bi = doc.extract_image(img[0])
                pil = Image.open(io.BytesIO(bi["image"]))
                if pil.width > 50 and pil.height > 50:
                    images.append({"page": pn+1, "image": pil, "index": ii+1, "size": f"{pil.width}×{pil.height}"})
        doc.close()
        os.unlink(tmp_path)
    except Exception as e:
        st.warning(f"Image error: {e}")
    return images


# ══════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛡️ Private AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-version">v3.0 — Enterprise Edition</div>', unsafe_allow_html=True)

    # Model selector
    if available_models:
        # Show all models but label vision models
        model_labels = []
        for m in available_models:
            if any(k in m for k in ["vision", "llava", "minicpm", "moondream", "gemma3"]):
                model_labels.append(f"{m} 👁️")
            else:
                model_labels.append(m)
        
        # Default to first non-vision model
        default_index = next((i for i, m in enumerate(available_models) if not any(k in m for k in ["vision", "llava", "minicpm", "moondream", "gemma3"])), 0)
        selected_label = st.selectbox("🤖 Active Model", options=model_labels, index=default_index)
        # Get actual model name (remove emoji label)
        selected_model = selected_label.replace(" 👁️", "")
    else:
        st.error("⚠️ Ollama not running!")
        selected_model = "No model"

    # Status
    is_cloud = ":cloud" in selected_model
    mode = "☁️ Cloud" if is_cloud else "🔒 Local"
    st.markdown(f'<span class="model-pill">{mode} · {selected_model}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if has_vision_model():
        st.markdown('<span class="status-ok">👁️ Vision Model Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">👁️ No Vision Model Found</span>', unsafe_allow_html=True)

    if OCR_AVAILABLE:
        st.markdown('<span class="status-ok">🔍 OCR Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">🔍 OCR Not Available</span>', unsafe_allow_html=True)

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "📄 Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Supports 5000+ PDFs simultaneously"
    )
    if uploaded_files:
        st.caption(f"📚 {len(uploaded_files)} file(s) selected")

    if st.button("⚡ Initialize Knowledge Base", use_container_width=True):
        if not available_models:
            st.error("Start Ollama first!")
        elif not uploaded_files:
            st.error("Upload at least one PDF!")
        else:
            all_docs = []
            total_scanned = 0
            progress = st.progress(0, text="Initializing...")
            total = len(uploaded_files)

            for i, file in enumerate(uploaded_files):
                progress.progress(i/total, text=f"Loading {file.name} ({i+1}/{total})")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getbuffer())
                    tmp_path = tmp.name
                try:
                    docs, scanned = smart_load_pdf(tmp_path, file.name)
                    all_docs.extend(docs)
                    total_scanned += scanned
                finally:
                    os.remove(tmp_path)

            progress.progress(0.8, text=f"Building FAISS index — {len(all_docs)} pages...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
            chunks = splitter.split_documents(all_docs)
            vector_db = FAISS.from_documents(chunks, st.session_state.embeddings)

            llm = ChatOllama(model=selected_model, temperature=0.3, num_predict=512)
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
                memory=st.session_state.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": custom_prompt}
            )
            progress.progress(1.0, text="✅ Ready!")
            ocr_msg = f" · {total_scanned} pages OCR'd" if total_scanned else ""
            st.success(f"✅ {len(uploaded_files)} PDF(s) · {len(chunks)} chunks{ocr_msg}")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="feature-desc" style="color:#334155">Built by Shobhit Singh<br>github.com/itskie</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════

# Hero header
st.markdown('<div class="hero-title">Private AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ENTERPRISE MULTI-PDF INTELLIGENCE PLATFORM · v3.0</div>', unsafe_allow_html=True)

# Badges
badges = ["FAISS Vector DB", "OCR Support", "Vision AI", "5000+ PDFs", "100% Private"]
badge_html = "".join([f'<span class="hero-badge">{b}</span>' for b in badges])
st.markdown(badge_html, unsafe_allow_html=True)

# ══════════════════════════════════════════
if uploaded_files:
    tab1, tab2, tab3, tab4 = st.tabs(["💬  Chat", "📊  Tables", "🖼️  Images & Vision", "📈  Charts"])

    # ── CHAT ──
    with tab1:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 Sources"):
                        for doc in msg["sources"]:
                            pg = doc.metadata.get("page", "?")
                            ocr = " · OCR" if doc.metadata.get("ocr") else ""
                            st.markdown(f'<span class="model-pill">Page {pg}{ocr}</span>', unsafe_allow_html=True)
                            st.caption(doc.page_content[:250] + "...")

        if prompt := st.chat_input("Ask anything about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.chain:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    with st.spinner(""):
                        result = st.session_state.chain.invoke({"question": prompt})
                        answer = result["answer"]
                        sources = result.get("source_documents", [])

                    words = answer.split(" ")
                    displayed = ""
                    for word in words:
                        displayed += word + " "
                        placeholder.markdown(displayed + "▌")
                        time.sleep(0.025)
                    placeholder.markdown(displayed)

                    if sources:
                        with st.expander("📎 Sources"):
                            for doc in sources:
                                pg = doc.metadata.get("page", "?")
                                ocr = " · OCR" if doc.metadata.get("ocr") else ""
                                st.markdown(f'<span class="model-pill">Page {pg}{ocr}</span>', unsafe_allow_html=True)
                                st.caption(doc.page_content[:250] + "...")

                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            else:
                st.info("⚡ Click **Initialize Knowledge Base** in the sidebar to begin.")

    # ── TABLES ──
    with tab2:
        st.markdown("### 📊 Extracted Tables")
        found = False
        for f in uploaded_files:
            tables = extract_tables(f)
            if tables:
                found = True
                st.markdown(f"**📄 {f.name}**")
                for t in tables:
                    st.markdown(f'<span class="model-pill">Page {t["page"]}</span>', unsafe_allow_html=True)
                    st.dataframe(t["df"], use_container_width=True)
                    st.download_button(
                        "⬇️ Export CSV",
                        t["df"].to_csv(index=False),
                        f"{f.name}_p{t['page']}.csv",
                        "text/csv",
                        key=f"csv_{f.name}_{t['page']}_{id(t['df'])}"
                    )
                    st.divider()
        if not found:
            st.info("No tables detected in uploaded PDFs.")

    # ── IMAGES & VISION ──
    with tab3:
        st.markdown("### 🖼️ Images & AI Vision Analysis")
        vision_ok = has_vision_model()
        if not vision_ok:
            st.warning("No vision model detected. Run: `ollama pull llama3.2-vision`")

        found = False
        for f in uploaded_files:
            imgs = extract_images(f)
            if imgs:
                found = True
                st.markdown(f"**📄 {f.name}** — {len(imgs)} image(s)")
                for img_data in imgs:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.image(img_data["image"],
                                caption=f"Page {img_data['page']} · {img_data['size']}",
                                use_container_width=True)
                    with c2:
                        if vision_ok:
                            q = st.text_input(
                                "Question for AI:",
                                value="Describe this. If it's a chart, explain the data and trends.",
                                key=f"vq_{f.name}_{img_data['page']}_{img_data['index']}_{id(img_data['image'])}"
                            )
                            if st.button("👁️ Analyze", key=f"vb_{f.name}_{img_data['page']}_{img_data['index']}_{id(img_data['image'])}"):
                                with st.spinner("Analyzing..."):
                                    result = analyze_image(img_data["image"], q)
                                st.markdown(f"**Analysis:**\n\n{result}")
                        else:
                            st.info("Install a vision model to analyze.")
                    st.divider()
        if not found:
            st.info("No images found in uploaded PDFs.")

    # ── CHARTS ──
    with tab4:
        st.markdown("### 📈 Interactive Chart Builder")
        all_tables = []
        for f in uploaded_files:
            for t in extract_tables(f):
                all_tables.append({"label": f"{f.name} · Page {t['page']}", "df": t["df"]})

        if not all_tables:
            st.info("No tables found. Upload PDFs with tabular data to generate charts.")
        else:
            sel = st.selectbox("Select table:", [t["label"] for t in all_tables])
            df = next(t["df"] for t in all_tables if t["label"] == sel)
            for col in df.columns:
                try: df[col] = pd.to_numeric(df[col])
                except: pass

            st.dataframe(df, use_container_width=True)
            num_cols = df.select_dtypes(include="number").columns.tolist()
            all_cols = df.columns.tolist()

            if len(all_cols) >= 2:
                c1, c2, c3 = st.columns(3)
                with c1: x = st.selectbox("X Axis", all_cols)
                with c2: y = st.selectbox("Y Axis", num_cols if num_cols else all_cols)
                with c3: ctype = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie", "Area"])

                if st.button("🎨 Generate Chart", use_container_width=True):
                    try:
                        fns = {"Bar": px.bar, "Line": px.line, "Scatter": px.scatter, "Area": px.area}
                        if ctype == "Pie":
                            fig = px.pie(df, names=x, values=y, title=f"{y} Distribution",
                                        color_discrete_sequence=px.colors.sequential.Blues_r)
                        else:
                            fig = fns[ctype](df, x=x, y=y, title=f"{y} · {ctype} Chart",
                                           color_discrete_sequence=["#60A5FA"])

                        fig.update_layout(
                            paper_bgcolor="#080C14",
                            plot_bgcolor="#0D1117",
                            font=dict(family="Syne", color="#E2E8F0"),
                            title_font=dict(size=16, color="#F1F5F9"),
                            xaxis=dict(gridcolor="#1E293B", color="#64748B"),
                            yaxis=dict(gridcolor="#1E293B", color="#64748B"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Chart error: {e}")

# ══════════════════════════════════════════
#  HOME SCREEN
# ══════════════════════════════════════════
else:
    features = [
        ("💬", "Smart Chat", "Ask questions across 5000+ PDFs simultaneously with conversation memory"),
        ("🔍", "OCR Engine", "Automatically detects and reads scanned or image-based PDFs"),
        ("👁️", "Vision AI", "Analyzes charts, diagrams and images using multimodal AI"),
        ("📊", "Table Extractor", "Auto-extracts tables from PDFs with one-click CSV export"),
        ("📈", "Chart Builder", "Transform PDF table data into interactive visualizations"),
        ("⚡", "FAISS Index", "Lightning-fast vector search — scales to millions of chunks"),
        ("🔒", "Zero Data Leaks", "100% local processing — your documents never leave your machine"),
        ("🌍", "Multilingual", "Responds in Hindi, English, or Hinglish automatically"),
    ]

    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Vector DB", "FAISS ⚡", "5000+ PDFs")
    with c2: st.metric("OCR", "pytesseract", "Scanned PDFs")
    vision_models = [m for m in available_models if any(k in m for k in ["vision", "llava", "minicpm", "moondream", "gemma3"])]
    vision_name = vision_models[0].split(":")[0] if vision_models else "Not installed"
    vision_delta = "Charts & Images" if vision_models else "💡 ollama pull llama3.2-vision"
    with c3: st.metric("Vision AI", vision_name, vision_delta)
    with c4: st.metric("Chunking", "600 tokens", "Optimized")