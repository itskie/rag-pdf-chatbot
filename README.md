# 🛡️ Private AI — Enterprise Multi-PDF Intelligence Platform

> A production-grade, fully local RAG system for intelligent document analysis. Chat with unlimited PDFs privately — no cloud, no data leaks, no compromises.

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.4-green?style=flat-square)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange?style=flat-square)](https://faiss.ai)
[![PageIndex](https://img.shields.io/badge/RAG-PageIndex-purple?style=flat-square)](https://github.com/VectifyAI/PageIndex)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [What's New in v4.0](#-whats-new-in-v40)
- [Features](#-features)
- [Architecture](#-architecture)
- [RAG Pipeline Deep Dive](#-rag-pipeline-deep-dive)
- [Auto Smart RAG](#-auto-smart-rag)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [How to Use](#-how-to-use)
- [Performance](#-performance)
- [System Requirements](#-system-requirements)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## 🔭 Overview

Private AI is a locally-running, enterprise-grade document intelligence platform built on a full RAG (Retrieval-Augmented Generation) pipeline. It enables users to upload any number of PDF documents and query them through a conversational AI interface — with zero data leaving the machine.

**The problem it solves:** Existing PDF chatbots either send your documents to external servers (privacy risk) or use naive vector similarity search that misses context, breaks paragraphs mid-sentence, and returns irrelevant chunks.

**How Private AI solves it:** A complete, production-quality RAG pipeline — semantic chunking, hybrid retrieval, cross-encoder reranking, and reasoning-based PageIndex retrieval — all running 100% locally via Ollama.

---

## 🆕 What's New in v4.0

| Version | What Changed |
|---|---|
| **v1.0** | Basic RAG chatbot with ChromaDB |
| **v2.0** | Real-time streaming, multilingual support, local + cloud model switching |
| **v3.0** | FAISS vector DB, OCR engine, Vision AI, table extraction, interactive charts |
| **v4.0** | PageIndex RAG, Auto Smart RAG, Hybrid Search (BM25 + FAISS), Semantic Chunking, CrossEncoder Reranker, modular codebase (6 files), real streaming |

---

## ✨ Features

| Feature | What | How | Why |
|---|---|---|---|
| 🧠 **Auto Smart RAG** | Automatically picks the best retrieval method | Counts pages on upload; ≤50 → PageIndex, >50 → FAISS | Eliminates manual choice; right tool for right document size |
| 🌲 **PageIndex RAG** | Reasoning-based document retrieval | Builds a hierarchical TOC tree; LLM reasons over summaries | Better than vector math for complex, multi-step questions |
| 🔀 **Hybrid Search** | Combines semantic + keyword retrieval | FAISS for meaning, BM25 for exact keywords; results merged | Catches what pure vector search misses (e.g. "Section 4.2") |
| 🎯 **Reranker** | Re-orders retrieved docs by true relevance | CrossEncoder reads question + doc together; scores all candidates | More accurate than FAISS similarity score alone |
| 🔍 **Semantic Chunking** | Splits documents at topic boundaries | SemanticChunker breaks on meaning shifts, not fixed token counts | Preserves context; related content stays in the same chunk |
| 💬 **Smart Chat** | Conversational Q&A over documents | Rolling 10-message window; real token-by-token streaming | Memory-safe; genuine live streaming (not fake word delay) |
| 🔍 **OCR Engine** | Reads scanned and image-based PDFs | pytesseract at 300 DPI auto-triggered on sparse pages | Works on any PDF, not just digital-native ones |
| 👁️ **Vision AI** | Analyzes images and charts inside PDFs | Extracts images via PyMuPDF; sends to Ollama vision model | Understands non-text content like graphs and diagrams |
| 📊 **Table Extractor** | Pulls tables from PDFs | pdfplumber per-page table detection; exports to CSV | Structured data becomes immediately usable |
| 📈 **Chart Builder** | Generates interactive charts from PDF tables | Plotly Express with axis and chart type selector | Visualize tabular data without leaving the app |
| 📚 **Source Citations** | Shows exact page used for each answer | Source documents returned from retriever; displayed in expander | Verifiable, traceable answers — no black-box retrieval |
| 🔒 **100% Private** | No data leaves the machine | All inference runs through local Ollama models | Enterprise-safe; suitable for confidential documents |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                        │
│         (main.py · sidebar.py · tabs.py)            │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Auto Smart RAG Router                   │
│     ≤ 50 pages → PageIndex   > 50 pages → FAISS     │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┴────────────────┐
       ▼                        ▼
┌─────────────┐        ┌────────────────────────────┐
│  PageIndex  │        │       FAISS Pipeline        │
│   (models)  │        │                            │
│             │        │  Semantic Chunking          │
│ Build TOC   │        │       ↓                    │
│ tree index  │        │  FAISS + BM25 Hybrid        │
│     ↓       │        │  Retrieval (top-6)          │
│ LLM reasons │        │       ↓                    │
│ over TOC    │        │  CrossEncoder Reranker      │
│     ↓       │        │  (top-3 selected)           │
│ Relevant    │        │       ↓                    │
│ pages       │        │  Relevant chunks            │
└──────┬──────┘        └────────────┬───────────────┘
       │                            │
       └────────────┬───────────────┘
                    ▼
        ┌───────────────────────┐
        │  chat_with_docs()     │
        │  StreamHandler →      │
        │  Ollama LLM           │
        │  (real streaming)     │
        └───────────┬───────────┘
                    ▼
            Answer + Sources
```

---

## 🔬 RAG Pipeline Deep Dive

### 1. Document Ingestion

**What:** PDFs are loaded page by page using PyPDFium2. Any page with fewer than 50 characters is identified as scanned and automatically processed through pytesseract OCR at 300 DPI.

**Why:** A single pipeline handles both digital-native and scanned PDFs without requiring user intervention.

### 2. Semantic Chunking

**What:** Documents are split at natural topic boundaries rather than fixed token counts.

**How:** `SemanticChunker` from LangChain Experimental embeds consecutive sentences and detects where meaning shifts by measuring embedding distance. When the distance crosses the 90th percentile threshold, a new chunk begins.

**Why:** Fixed-size chunking (e.g. 600 tokens) often splits mid-paragraph, destroying context. Semantic chunking keeps related content together — a paragraph discussing Q3 revenue stays in one chunk even if it spans 800 tokens.

```
Fixed chunking (old):      Semantic chunking (new):
"...revenue was $4.2M      "Revenue section → one chunk
 The risk facto-"          Risk factors → separate chunk"
```

### 3. Hybrid Retrieval

**What:** Combines FAISS vector search with BM25 keyword search.

**How:**
- FAISS performs approximate nearest-neighbour search using cosine similarity on embedded vectors (top-6 results)
- BM25 scores all chunks using TF-IDF-style keyword matching (top-6 results)
- Both result sets are merged and deduplicated by content fingerprint

**Why:** Semantic search alone misses exact terms. Searching for "Section 4.2 EBITDA" semantically returns revenue-adjacent content, but BM25 catches the exact string "Section 4.2". Hybrid retrieval combines both signals.

### 4. CrossEncoder Reranking

**What:** A CrossEncoder model re-scores the merged candidate documents and selects the top 3.

**How:** Unlike bi-encoder models (FAISS) that embed query and document separately, a CrossEncoder concatenates them and processes both simultaneously through a transformer. This produces a more accurate relevance score.

**Why:** FAISS similarity score ≠ actual relevance. A chunk about "annual revenue" may score high for "what is the company's risk exposure?" simply because both documents are financial. The CrossEncoder reads the full context and correctly down-ranks irrelevant chunks.

```
Pipeline:
PDF chunks → Hybrid Retrieve (top 6) → CrossEncoder → Top 3 → LLM
```

### 5. PageIndex Retrieval

**What:** An alternative retrieval mode that builds a hierarchical document tree and uses LLM reasoning to navigate it.

**How:**
1. **Build phase:** Every 5 pages are grouped into a node. An LLM generates a title and 2-sentence summary for each node.
2. **Retrieve phase:** The full table of contents is presented to the LLM with the user's question. The LLM reasons which sections likely contain the answer and returns section numbers.

**Why:** Inspired by [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex). For short, dense documents (legal contracts, research papers), vector similarity misses multi-step reasoning. PageIndex lets the LLM think like a human expert navigating a document — reading the index, not scanning every word.

### 6. Streaming Response

**What:** Token-by-token streaming via a LangChain `BaseCallbackHandler`.

**How:** `StreamHandler.on_llm_new_token()` is called for every token emitted by the LLM. Each token is appended to a Streamlit placeholder with a blinking cursor (`▌`). On `on_llm_end()`, the cursor is removed.

**Why:** The previous v3.0 implementation faked streaming — the full response was generated internally and then displayed word-by-word with `time.sleep()`. This added unnecessary latency. Real streaming starts displaying output the moment the first token is generated.

---

## 🧠 Auto Smart RAG

**What:** Automatically selects FAISS or PageIndex based on document size — no manual choice required.

**How:**

```
Upload PDFs → Count total pages
                    │
          ┌─────────┴──────────┐
       ≤ 50 pages          > 50 pages
          │                    │
    PageIndex              FAISS + Hybrid
    (reasoning)            + Reranker
```

**Why:** PageIndex requires one LLM call per document section during indexing. For a 500-page document, this takes 15–20 minutes and provides marginal benefit over FAISS. For a 20-page legal contract, PageIndex's reasoning significantly outperforms vector similarity. The threshold of 50 pages was chosen as the practical crossover point.

---

## 🏗️ Tech Stack

| Layer | Component | Technology | Purpose |
|---|---|---|---|
| UI | Frontend | Streamlit | Rapid, reactive web interface |
| LLM | Inference | Ollama | Local model serving |
| Embeddings | Encoding | HuggingFace `all-MiniLM-L6-v2` | Lightweight, fast document embeddings |
| Vector DB | Storage | FAISS (CPU) | Approximate nearest-neighbour search |
| Keyword Search | Retrieval | rank-bm25 | TF-IDF keyword matching |
| Reranker | Scoring | CrossEncoder `ms-marco-MiniLM-L-6-v2` | True relevance scoring |
| Chunking | Splitting | LangChain SemanticChunker | Topic-aware document splitting |
| RAG | Reasoning | PageIndex (custom) | Hierarchical tree-based retrieval |
| PDF | Extraction | PyPDFium2 | Fast, accurate text extraction |
| Tables | Extraction | pdfplumber | Structured table detection |
| Images | Extraction | PyMuPDF (fitz) | Image extraction from PDF pages |
| OCR | Recognition | pytesseract + Tesseract | Scanned page text recognition |
| Charts | Visualization | Plotly Express | Interactive data charts |
| Vision | Multimodal | llama3.2-vision via Ollama | Image and chart analysis |

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── main.py              # Entry point — session state, layout, routing
├── config.py            # Page config and CSS theme injection
├── models.py            # StreamHandler, PageIndexTree classes
├── utils.py             # PDF loading, chunking, retrieval, reranking, chat
├── sidebar.py           # Sidebar UI, KB initialization, Auto Smart RAG logic
├── tabs.py              # Chat, Document Index, Tables, Vision, Charts tabs
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

Each file has a single responsibility. The modular structure replaces the original monolithic `app.py` (782 lines) with six focused modules averaging ~200 lines each.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/itskie/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Create a virtual environment

```bash
# Mac / Linux
python3.12 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
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
- Download from [ollama.com](https://ollama.com) and run the installer

### 5. Pull a model

```bash
# Recommended — fast, cloud inference, no download
ollama run qwen3.5:cloud

# Local only — full privacy
ollama pull llama3.1          # 4.9 GB — chat
ollama pull llama3.2-vision   # 7.8 GB — chat + vision
```

### 6. Install Tesseract OCR *(optional — only for scanned PDFs)*

**Mac:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt install tesseract-ocr
```

**Windows:**
- Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- Add `C:\Program Files\Tesseract-OCR` to PATH

### 7. Run

```bash
streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 How to Use

1. **Upload PDFs** — drag and drop one or more PDFs into the sidebar
2. **Initialize** — click ⚡ Initialize Knowledge Base
   - Auto Smart RAG automatically selects the retrieval mode
   - Progress bar shows chunking and indexing status
3. **Chat** — ask questions in any language (English, Hindi, Hinglish, etc.)
4. **Inspect sources** — expand 📎 Sources under any answer to see the exact pages used
5. **Reasoning trace** — if PageIndex is active, expand 🧠 Reasoning Trace to see the LLM's navigation logic
6. **Tables** — switch to the Tables tab to extract and download structured data as CSV
7. **Vision** — switch to Vision tab to analyze charts and images using the multimodal model
8. **Charts** — build interactive charts from any extracted table

---

## ⚡ Performance

| Optimization | Value | Reason |
|---|---|---|
| Semantic chunk size | Variable (topic-based) | Preserves context better than fixed 600 tokens |
| Hybrid retrieval k | 6 candidates | Wide net before reranking |
| Reranker top-k | 3 documents | Sufficient context without overwhelming the LLM |
| LLM temperature | 0.3 | Focused, deterministic answers |
| Max tokens | 512 | Prevents slow, overly verbose responses |
| Embeddings | Cached per session | Loaded once, reused across all queries |
| Table extraction | `@st.cache_data` | Tables tab and Charts tab share one extraction pass |
| Ollama models | `@st.cache_data(ttl=30)` | HTTP call made at most once per 30 seconds |
| Reranker model | `@st.cache_resource` | CrossEncoder loaded once per session |

---

## ⚙️ System Requirements

| Spec | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Storage | 15 GB | 25 GB |
| Python | 3.10+ | 3.12 |
| OS | Mac / Linux / Windows | Mac M1+ / Linux |

---

## 🔧 Troubleshooting

**Ollama not running?**
```bash
ollama serve          # Linux / Windows
brew services start ollama   # Mac
```

**Model not in dropdown?**
```bash
ollama list           # Check installed models
ollama pull llama3.1  # Pull if missing
```

**Tesseract not found?**
```bash
brew install tesseract        # Mac
sudo apt install tesseract-ocr  # Linux
```

**Port already in use?**
```bash
streamlit run main.py --server.port 8502
```

**Out of memory?**
- Use `qwen3.5:cloud` — runs on provider servers, zero local RAM
- Use `llama3.2-vision` only in the Vision tab

**Semantic chunking slow?**
- Falls back to `RecursiveCharacterTextSplitter` automatically if `langchain-experimental` fails
- No manual action needed

---

## 🗺️ Roadmap

- [x] v1.0 — Basic RAG chatbot with ChromaDB
- [x] v2.0 — Real-time streaming, multilingual, local + cloud models
- [x] v3.0 — FAISS, OCR, Vision AI, tables, charts, 5000+ PDF support
- [x] v4.0 — PageIndex RAG, Auto Smart RAG, Hybrid Search, Semantic Chunking, Reranker, modular codebase
- [ ] v5.0 — FastAPI backend, Docker support

---

## 👨‍💻 Author

**Shobhit Kumar Singh**

[![GitHub](https://img.shields.io/badge/GitHub-itskie-black?style=flat-square&logo=github)](https://github.com/itskie)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-itskie-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/itskie)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built with ❤️ by Shobhit Kumar Singh</p>
