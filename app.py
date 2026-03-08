import streamlit as st
import os
import requests
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate

# --- Page Config ---
st.set_page_config(page_title="Private AI", page_icon="🛡️", layout="wide")
st.title("🛡️ Private AI — Local & Cloud Multi-PDF Chatbot")

# --- Auto Detect Ollama Models ---
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

# --- Initialize Session States ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

# --- Custom Prompt ---
custom_prompt = PromptTemplate(
    template="""You are a highly intelligent and friendly AI assistant.
Answer the user's question clearly and conversationally. Be concise but complete.

Rules:
- Give a helpful, direct answer
- Use simple language
- Do NOT say "based on the context" or "according to the document"
- Answer naturally and confidently
- If you don't know something, say so honestly
- VERY IMPORTANT: Always respond in the SAME language as the user's question.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:""",
    input_variables=["context", "chat_history", "question"]
)

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Documents")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()

    if available_models:
        selected_model = st.selectbox(
            "🤖 Detected Models",
            options=available_models,
            index=0
        )
    else:
        st.error("⚠️ Ollama is not running or no models found. Please start Ollama.")
        selected_model = "No model detected"

    st.divider()

    uploaded_files = st.file_uploader("Select PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Initialize Knowledge Base"):
        if available_models and uploaded_files:
            with st.spinner("Processing documents. Please wait..."):
                all_docs = []
                for file in uploaded_files:
                    with open(file.name, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFium2Loader(file.name)
                    all_docs.extend(loader.load())
                    os.remove(file.name)

                # Optimized chunking
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=80
                )
                chunks = splitter.split_documents(all_docs)

                # Cache embeddings
                if "embeddings" not in st.session_state:
                    st.session_state.embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2"
                    )

                vector_db = Chroma.from_documents(chunks, st.session_state.embeddings)

                # ChatOllama = real streaming support
                llm = ChatOllama(
                    model=selected_model,
                    temperature=0.3,
                    num_predict=512,
                )

                st.session_state.chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vector_db.as_retriever(
                        search_kwargs={"k": 3}
                    ),
                    memory=st.session_state.memory,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": custom_prompt}
                )
                st.success(f"✅ Loaded {len(uploaded_files)} doc(s) with {selected_model} ⚡")
        elif not available_models:
            st.error("Please start Ollama first.")
        else:
            st.error("Please upload at least one PDF file before initializing.")

# --- Dynamic Subtitle ---
is_cloud = ":cloud" in selected_model
mode_label = "☁️ Cloud" if is_cloud else "🔒 Local"
st.markdown(f"Built by Shobhit Singh | {mode_label} mode — running on **{selected_model}**")

# --- Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Sources"):
                for doc in msg["sources"]:
                    st.write(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                    st.caption(doc.page_content[:250] + "...")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "chain" in st.session_state:
        with st.chat_message("assistant"):
            # Real GPT-style streaming
            response_placeholder = st.empty()
            full_answer = ""

            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke({"question": prompt})
                full_answer = response['answer']
                sources = response.get('source_documents', [])

            # Stream token by token — real streaming display
            def stream_response(text):
                import time
                words = text.split(" ")
                displayed = ""
                for word in words:
                    displayed += word + " "
                    response_placeholder.markdown(displayed + "▌")
                    time.sleep(0.03)  # natural typing speed
                response_placeholder.markdown(displayed)

            stream_response(full_answer)

            if sources:
                with st.expander("📚 View Sources"):
                    for doc in sources:
                        st.write(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                        st.caption(doc.page_content[:250] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources
            })
    else:
        st.info("Please initialize the knowledge base from the sidebar before asking questions.")
