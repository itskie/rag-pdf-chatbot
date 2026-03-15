import re
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_ollama import ChatOllama


class StreamHandler(BaseCallbackHandler):
    """
    Real token-by-token streaming handler for Streamlit.

    Intercepts each token from the LLM as it is generated
    and writes it live to a Streamlit placeholder element.
    """

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, response, **kwargs):
        self.container.markdown(self.text)


class PageIndexTree:
    """
    Reasoning-based RAG inspired by VectifyAI/PageIndex.

    Traditional FAISS flow:
        Question -> vectors -> cosine similarity -> chunks

    PageIndex flow:
        Build:    Pages -> batch groups -> LLM summary -> tree nodes
        Retrieve: Question -> LLM reads TOC -> reasons -> relevant pages

    Advantages over vector search:
        - Understands meaning, not just keyword similarity
        - Explainable retrieval with reasoning trace
        - No vector math required at query time
    """

    def __init__(self, docs: list, model_name: str, batch_size: int = 5):
        self.docs = docs
        self.model_name = model_name
        self.batch_size = batch_size
        self.tree = []
        self.built = False

    def build(self, progress_callback=None):
        """Build the hierarchical tree index from documents."""
        llm = ChatOllama(model=self.model_name, temperature=0)
        total_batches = (len(self.docs) + self.batch_size - 1) // self.batch_size

        for i, batch_start in enumerate(range(0, len(self.docs), self.batch_size)):
            batch = self.docs[batch_start: batch_start + self.batch_size]
            pages = [d.metadata.get("page", batch_start + j) for j, d in enumerate(batch)]

            section_text = "\n\n".join([
                f"[Page {pages[j]}]\n{d.page_content[:600]}"
                for j, d in enumerate(batch)
            ])

            prompt = (
                "Analyze this document section. Provide:\n"
                "1. TITLE: A short title (5 words max)\n"
                "2. SUMMARY: What topics does this section cover? (2 sentences)\n\n"
                f"Section:\n{section_text}\n\n"
                "Respond EXACTLY in this format:\n"
                "TITLE: <title>\n"
                "SUMMARY: <summary>"
            )

            try:
                response = llm.invoke(prompt)
                raw = response.content if hasattr(response, "content") else str(response)
                title_m = re.search(r"TITLE:\s*(.+)", raw)
                summary_m = re.search(r"SUMMARY:\s*(.+)", raw, re.DOTALL)
                title = title_m.group(1).strip() if title_m else f"Section {i + 1}"
                summary = summary_m.group(1).strip()[:200] if summary_m else raw[:200]
            except Exception:
                title = f"Pages {pages[0]}-{pages[-1]}"
                summary = section_text[:150]

            self.tree.append({
                "id": i + 1,
                "title": title,
                "pages": pages,
                "start_page": pages[0],
                "end_page": pages[-1],
                "summary": summary,
                "docs": batch,
            })

            if progress_callback:
                progress_callback((i + 1) / total_batches)

        self.built = True

    def retrieve(self, question: str, chat_history: list = None, k: int = 2):
        """
        Retrieve relevant documents using LLM reasoning over the tree index.

        Returns:
            tuple: (relevant_docs, reasoning_trace)
        """
        if not self.built or not self.tree:
            return [], "Tree not built yet."

        llm = ChatOllama(model=self.model_name, temperature=0)

        toc = "\n".join([
            f"[Section {n['id']}] Pages {n['start_page']}-{n['end_page']} | {n['title']}\n"
            f"  Summary: {n['summary']}"
            for n in self.tree
        ])

        history_ctx = ""
        if chat_history:
            recent = chat_history[-4:]
            history_ctx = "\nRecent conversation:\n" + "\n".join([
                f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:100]}"
                for m in recent
            ])

        reasoning_prompt = (
            "You are navigating a document index to find an answer.\n"
            "Read the sections and identify which ones contain the answer.\n\n"
            f"Document Index:\n{toc}"
            f"{history_ctx}\n\n"
            f"Question: {question}\n\n"
            "Reply with ONLY comma-separated section numbers, most relevant first.\n"
            "Example: 3,1,5\n\n"
            "Most relevant sections:"
        )

        relevant_docs = []
        reasoning_trace = ""

        try:
            response = llm.invoke(reasoning_prompt)
            reasoning_trace = response.content if hasattr(response, "content") else str(response)

            numbers = re.findall(r"\d+", reasoning_trace)
            selected_ids = []
            for n in numbers:
                idx = int(n)
                if 1 <= idx <= len(self.tree) and idx not in selected_ids:
                    selected_ids.append(idx)
                if len(selected_ids) >= k:
                    break

            for sid in selected_ids:
                relevant_docs.extend(self.tree[sid - 1]["docs"])

        except Exception as e:
            for node in self.tree[:k]:
                relevant_docs.extend(node["docs"])
            reasoning_trace = f"Fallback retrieval. Error: {e}"

        return relevant_docs, reasoning_trace
