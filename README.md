# 🛡️ Private AI — Local Multi-PDF Chatbot

A fully private, locally-running RAG (Retrieval-Augmented Generation) chatbot that lets you chat with your PDF documents. No data leaves your machine.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Classic-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 📄 **Multi-PDF Support** — Upload and query multiple PDFs simultaneously
- 🔒 **100% Private** — All processing happens locally, no data sent to the cloud
- 🤖 **Auto Model Detection** — Automatically detects all installed Ollama models
- 🌍 **Multilingual** — Responds in the same language you ask (Hindi, English, Hinglish, Chinese, etc.)
- 📚 **Source Citations** — View the exact page and content that was used to generate the answer
- 🧠 **Conversation Memory** — Remembers chat history for follow-up questions
- 🗑️ **Clear Chat** — Reset conversation anytime

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Ollama (Llama 3.1, Mistral, Qwen, etc.) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| RAG Framework | LangChain Classic |
| PDF Loader | PyPDFium2 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- At least one Ollama model pulled

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/iTsKie/Private-AI-Bot.git
cd Private-AI-Bot
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start Ollama and pull a model**
```bash
brew services start ollama        # macOS
ollama pull llama3.1              # or any model you prefer
```

**5. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📖 Usage

1. Upload one or more PDF files from the sidebar
2. Select your preferred LLM model from the dropdown
3. Click **"Initialize Knowledge Base"**
4. Start chatting with your documents!

---

## 📁 Project Structure

```
Private-AI-Bot/
├── app.py              # Main application
├── requirements.txt    # Project dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

---

## 🔮 Roadmap

- [ ] Support for DOCX and TXT files
- [ ] Persistent vector database across sessions
- [ ] OpenAI / Claude / Gemini API integration
- [ ] Docker support for easy deployment
- [ ] Chat history export

---

## 👨‍💻 Author

**Shobhit Kumar Singh**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/itskie)

---

## 📄 License

This project is licensed under the MIT License.
