# 🛡️ Private AI — Enterprise Multi-PDF Chatbot

> Chat with unlimited PDFs privately. 100% local. No data leaks.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.4-green?style=flat-square)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange?style=flat-square)
![Ollama](https://img.shields.io/badge/LLM-Ollama-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Smart Chat** | Ask questions across 5000+ PDFs with conversation memory |
| 📊 **Table Extractor** | Auto-extract tables from PDFs with CSV export |
| 🖼️ **Vision AI** | Analyze images & charts inside PDFs using multimodal AI |
| 📈 **Chart Builder** | Generate interactive charts from PDF table data |
| 🔍 **OCR Engine** | Auto-reads scanned/image-based PDFs |
| ⚡ **FAISS Index** | Lightning-fast vector search — scales to millions of chunks |
| 🔀 **Hybrid Inference** | Switch between local (private) and cloud (fast) models |
| 🤖 **Auto Model Detection** | Automatically detects all installed Ollama models |
| ⚡ **Real-time Streaming** | ChatGPT-style word-by-word response |
| 🌍 **Multilingual** | Responds in Hindi, English, Hinglish, Chinese, Spanish, French, Arabic & more |
| 📚 **Source Citations** | View exact page and content used for each answer |
| 🔒 **100% Private** | Your documents never leave your machine |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│        Your App (Streamlit UI)       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│              Ollama                  │
│        (Universal Interface)         │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────────┐
│ Local Models │  │  Cloud Models    │
│  (Llama3.2  │  │ (qwen3.5:cloud,  │
│   Mistral,  │  │  minimax:cloud,  │
│   etc.)     │  │  glm-5:cloud..)  │
│             │  │                  │
│ ✅ 100%     │  │ ⚡ Zero Download │
│   Private   │  │ ❄️ No GPU Load   │
│ 🖥️ On your  │  │ 🌐 Runs on their │
│   machine   │  │   servers        │
└─────────────┘  └──────────────────┘
```

> **How it works:** Your app talks only to Ollama. Ollama handles routing — local inference on your machine or cloud inference on remote servers. The app code stays identical either way.

---

## ⚖️ Local vs Cloud — When to Use What

| Feature | 🔒 Local Models | ☁️ Cloud Models |
|---|---|---|
| **Privacy** | 100% — data never leaves device | Data sent to model provider |
| **Download Required** | 4GB–8GB per model | ❌ Zero |
| **MacBook Load** | High (GPU/CPU) | ❌ None |
| **Speed** | Medium | Fast |
| **Cost** | Free forever | May have usage limits |
| **Best For** | Sensitive docs, offline use | Quick queries, large models |

---

## ⚡ Performance Optimizations

| Optimization | Value | Reason |
|---|---|---|
| **Chunk Size** | 600 tokens | Best balance of speed & context quality |
| **Chunk Overlap** | 80 tokens | Reduces redundancy, keeps context flow |
| **Top-K Retrieval** | 3 chunks | Less context = faster LLM response |
| **LLM Temperature** | 0.3 | Focused, faster, less hallucination |
| **Max Tokens** | 512 | Prevents overly long slow responses |
| **Embeddings** | Cached per session | Loads only once, reused on re-init |
| **Vector DB** | FAISS | 10x faster than ChromaDB at scale |
| **Streaming** | Real-time word-by-word | ChatGPT-style live cursor display |

---

## 🧠 How RAG Works (Under the Hood)

```
Your PDFs
   ↓
[Text Extraction] — PyPDFium2 reads every page
   ↓
[OCR] — pytesseract reads scanned pages automatically
   ↓
[Chunking] — Text split into 600-token overlapping chunks
   ↓
[Embedding] — HuggingFace converts chunks to vectors (cached)
   ↓
[Vector Store] — FAISS stores all vectors (5000+ PDF support)
   ↓
Your Question
   ↓
[Similarity Search] — Find top-3 most relevant chunks
   ↓
[Context + Question] → sent to LLM (local or cloud)
   ↓
[Streaming Response] — Word by word display ⚡
   ↓
Answer with source citations ✅
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/itskie/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Create virtual environment

```bash
# Mac / Linux
python3.12 -m venv venv312
source venv312/bin/activate

# Windows
python -m venv venv312
venv312\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

**Mac:**
```bash
brew install ollama
brew services start ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

**Windows:**
- Download installer from [ollama.com](https://ollama.com)
- Run the `.exe` installer
- Ollama starts automatically

### 5. Pull AI Models

**Recommended setup (8GB RAM):**

```bash
# Chat model — cloud (zero download, fast)
ollama run qwen3.5:cloud

# Vision model — for image/chart analysis
ollama pull llama3.2-vision
```

**Single model setup (16GB+ RAM):**

```bash
ollama pull llama3.2-vision  # handles both chat + vision
```

**Local only setup (full privacy):**

```bash
ollama pull llama3.1         # 4.9GB — chat only
ollama pull llama3.2-vision  # 7.8GB — vision
```

### 6. Install Tesseract OCR *(for scanned PDFs)*

**Mac:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install tesseract
```

**Windows:**
- Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- Run `.exe` installer
- Add to PATH: `C:\Program Files\Tesseract-OCR`
- Verify: `tesseract --version`

> ⚠️ Tesseract is **optional** — only needed for scanned/image-based PDFs. Normal text PDFs work without it.

### 7. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 How to Use

1. **Upload PDFs** — drag & drop in the sidebar (supports multiple files)
2. **Select Model** — choose your Ollama model from dropdown
   - `qwen3.5:cloud` → fast, zero download, cloud inference
   - `llama3.2-vision` 👁️ → local, vision capable, full privacy
3. **Initialize** — click ⚡ Initialize Knowledge Base
4. **Chat** — ask in any language — Hindi, English, Hinglish, Chinese, Spanish, French, Arabic & more
5. **Tables** — extract & download tables as CSV
6. **Images & Vision** — analyze diagrams and charts with AI
7. **Charts** — build interactive charts from PDF data

---

## 🔒 Privacy Notice

> **Local models** (e.g., `llama3.1`, `llama3.2-vision`): Your documents and queries are processed **entirely on your machine**. Nothing is sent to any external server.
>
> **Cloud models** (e.g., `qwen3.5:cloud`): Your queries and relevant document chunks are sent to the **model provider's servers** for inference. Do not use cloud models with sensitive or confidential documents.

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Ollama (qwen3.5, llama3.2-vision, or any local model) |
| Vision AI | llama3.2-vision via Ollama |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| PDF Loader | PyPDFium2 |
| Table Extract | pdfplumber |
| Image Extract | PyMuPDF (fitz) |
| OCR | pytesseract + tesseract |
| Charts | Plotly Express |
| Memory | LangChain ConversationBufferMemory |
| Chain | LangChain ConversationalRetrievalChain |

---

## ⚙️ System Requirements

| Spec | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Storage | 15 GB | 20 GB |
| Python | 3.10+ | 3.12 |
| OS | Mac / Linux / Windows | Mac M1+ |

---

## 🔧 Troubleshooting

**Ollama not running?**
```bash
# Mac
brew services start ollama

# Linux
ollama serve

# Windows — open Ollama from Start Menu or run:
ollama serve
```

**Model not showing in dropdown?**
```bash
ollama list                 # Check installed models
ollama pull qwen3.5:cloud   # Pull if missing
```

**Tesseract not found?**
```bash
# Mac
brew install tesseract

# Linux
sudo apt install tesseract-ocr

# Windows — add to PATH:
# C:\Program Files\Tesseract-OCR
# then restart terminal
```

**pip install fails on Linux?**
```bash
sudo apt install python3-pip python3-dev
pip install -r requirements.txt
```

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Out of memory?**
- Use `qwen3.5:cloud` for chat (runs in cloud, saves RAM)
- Use `llama3.2-vision` only for Images tab

**Wrong Python environment?**
```bash
# Mac/Linux
source venv312/bin/activate

# Windows
venv312\Scripts\activate
```

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

---

## 🗺️ Roadmap

- [x] v1.0 — Basic RAG chatbot with ChromaDB
- [x] v2.0 — Streaming, multilingual, local + cloud modes
- [x] v3.0 — FAISS, OCR, Vision AI, Tables, Charts, 5000+ PDFs
- [ ] v4.0 — Web URL ingestion, API endpoint, Docker support

---

## 👨‍💻 Author

**Shobhit Kumar Singh**

[![GitHub](https://img.shields.io/badge/GitHub-itskie-black?style=flat-square&logo=github)](https://github.com/itskie)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-itskie-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/itskie)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built with ❤️ by Shobhit Singh</p>
