# 🛡️ Private AI — Local & Cloud Multi-PDF Chatbot

A powerful, flexible RAG (Retrieval-Augmented Generation) chatbot that lets you chat with your PDF documents. Choose between **100% local privacy** or **cloud-powered speed** — all through a single unified interface.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Classic-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20%26%20Cloud-black)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 📄 **Multi-PDF Support** — Upload and query multiple PDFs simultaneously
- 🔀 **Hybrid Inference** — Switch between local models (private) and cloud models (fast) via Ollama
- 🔒 **100% Private Mode** — Use local models and no data ever leaves your machine
- ☁️ **Cloud Mode** — Use Ollama's cloud inference (minimax-m2.5:cloud, qwen3.5:cloud, etc.) with zero download
- 🤖 **Auto Model Detection** — Automatically detects all installed and available Ollama models
- ⚡ **Real-time Streaming** — ChatGPT-style word-by-word response with live cursor
- 🌍 **Multilingual** — Responds in the same language you ask (Hindi, English, Hinglish, Chinese, etc.)
- 📚 **Source Citations** — View the exact page and content used to generate each answer
- 🧠 **Conversation Memory** — Remembers chat history for natural follow-up questions
- 💾 **Cached Embeddings** — Embedding model loads only once per session for faster re-initialization
- 🗑️ **Clear Chat** — Reset conversation anytime

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
│  (Llama3.1  │  │ (minimax:cloud,  │
│   Mistral,  │  │  qwen3.5:cloud,  │
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
| **Download Required** | 8GB–25GB per model | ❌ Zero |
| **MacBook Load** | High (GPU/CPU) | ❌ None |
| **Speed** | Medium | Fast |
| **Cost** | Free forever | May have usage limits |
| **Best For** | Sensitive docs, offline use | Quick queries, large models |

---

## ⚡ Performance Optimizations

This app is built with speed in mind. Here's what's optimized under the hood:

| Optimization | Value | Reason |
|---|---|---|
| **Chunk Size** | 600 tokens | Best balance of speed & context quality |
| **Chunk Overlap** | 80 tokens | Reduces redundancy, keeps context flow |
| **Top-K Retrieval** | 3 chunks | Less context = faster LLM response |
| **LLM Temperature** | 0.3 | Focused, faster, less hallucination |
| **Max Tokens** | 512 | Prevents overly long slow responses |
| **Embeddings** | Cached per session | Loads only once, reused on re-init |
| **Streaming** | Real-time word-by-word | ChatGPT-style live cursor display |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM Interface | Ollama (Local + Cloud Inference) |
| LLM Class | ChatOllama (streaming support) |
| Local Models | Llama 3.1, Mistral, Qwen, etc. |
| Cloud Models | minimax-m2.5:cloud, qwen3.5:cloud, glm-5:cloud, kimi-k2.5:cloud |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (cached) |
| Vector Store | ChromaDB |
| RAG Framework | LangChain Classic |
| PDF Loader | PyPDFium2 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running (v0.5+ recommended for cloud model support)
- At least one model available (local or cloud)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/itskie/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

**2. Create and activate virtual environment**
```bash
python3.12 -m venv venv312
source venv312/bin/activate  # On Windows: venv312\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start Ollama**
```bash
brew services start ollama        # macOS
```

**5. Pull a model (choose your mode)**

```bash
# Option A — Local Model (private, runs on your machine)
ollama pull llama3.1              # ~4.7GB

# Option B — Cloud Model (no download needed, Ollama 0.5+)
ollama run minimax-m2.5:cloud     # Zero download, cloud inference
ollama run qwen3.5:cloud          # Zero download, cloud inference
```

**6. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📖 Usage

1. Upload one or more PDF files from the sidebar
2. Select your preferred model from the dropdown
   - Local models → full privacy, runs on your device
   - `:cloud` models → faster, zero download, runs on provider servers
3. Click **"Initialize Knowledge Base"**
4. Start chatting with your documents!

> 💡 **Tip:** For faster responses, use smaller PDFs (under 5MB). Large PDFs (20MB+) take longer to process and retrieve from.

---

## 🔒 Privacy Notice

> **Local models** (e.g., `llama3.1`, `mistral`): Your documents and queries are processed **entirely on your machine**. Nothing is sent to any external server.
>
> **Cloud models** (e.g., `minimax-m2.5:cloud`, `qwen3.5:cloud`): Your queries and relevant document chunks are sent to the **model provider's servers** for inference. Do not use cloud models with sensitive or confidential documents.

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py              # Main application
├── requirements.txt    # Project dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

---

## 🧠 How RAG Works (Under the Hood)

```
Your PDFs
   ↓
[Text Extraction] — PyPDFium2 reads every page
   ↓
[Chunking] — Text split into 600-token overlapping chunks
   ↓
[Embedding] — HuggingFace converts chunks to vectors (cached)
   ↓
[Vector Store] — ChromaDB stores all vectors
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

## 🔮 Roadmap

- [ ] Support for DOCX and TXT files
- [ ] Persistent vector database across sessions
- [ ] OpenAI / Claude / Gemini API integration
- [ ] Docker support for easy deployment
- [ ] Chat history export
- [ ] Model performance comparison dashboard

---

## 👨‍💻 Author

**Shobhit Singh**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/itskie)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/itskie)

---

## 📄 License

This project is licensed under the MIT License.
