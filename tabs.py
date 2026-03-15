import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    extract_tables,
    extract_images,
    analyze_image,
    chat_with_docs,
    has_vision_model,
    hybrid_retrieve,
    rerank_docs,
)


def render_tabs(uploaded_files, selected_model, available_models):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬  Chat",
        "🌲  Document Index",
        "📊  Tables",
        "🖼️  Vision",
        "📈  Charts",
    ])
    with tab1:
        render_chat_tab(selected_model)
    with tab2:
        render_index_tab()
    with tab3:
        render_tables_tab(uploaded_files)
    with tab4:
        render_vision_tab(uploaded_files, available_models)
    with tab5:
        render_charts_tab(uploaded_files)


def render_chat_tab(selected_model):
    # Show active retrieval mode
    if st.session_state.kb_initialized:
        mode = st.session_state.rag_mode
        if mode == "pageindex":
            st.markdown(
                '<span class="model-pill" style="border-color:rgba(167,139,250,0.4);color:#A78BFA">'
                '🌲 PageIndex — Reasoning-Based · Auto-selected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="model-pill">⚡ FAISS — Hybrid Search + Reranker · Auto-selected</span>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for doc in msg["sources"]:
                        pg      = doc.metadata.get("page", "?")
                        ocr_tag = " · OCR" if doc.metadata.get("ocr") else ""
                        st.markdown(
                            f'<span class="model-pill">Page {pg}{ocr_tag}</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(doc.page_content[:250] + "...")
            if msg.get("reasoning"):
                with st.expander("🧠 Reasoning Trace"):
                    st.code(msg["reasoning"], language=None)

    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        kb_ready = (
            (st.session_state.rag_mode == "faiss" and st.session_state.vector_db is not None)
            or (st.session_state.rag_mode == "pageindex"
                and st.session_state.page_tree is not None
                and st.session_state.page_tree.built)
        )

        if kb_ready:
            with st.chat_message("assistant"):
                placeholder    = st.empty()
                reasoning_trace = None

                if st.session_state.rag_mode == "faiss":
                    # Hybrid retrieval: FAISS + BM25, then rerank
                    broad_docs   = hybrid_retrieve(
                        question   = prompt,
                        vector_db  = st.session_state.vector_db,
                        all_chunks = st.session_state.get("all_chunks", []),
                        k          = 6,
                    )
                    context_docs = rerank_docs(prompt, broad_docs, top_k=3)
                else:
                    # PageIndex: LLM reasons over document tree
                    with st.spinner("🌲 Reasoning through document tree..."):
                        context_docs, reasoning_trace = st.session_state.page_tree.retrieve(
                            question     = prompt,
                            chat_history = st.session_state.messages,
                            k            = 2,
                        )

                answer = chat_with_docs(
                    question     = prompt,
                    context_docs = context_docs,
                    messages     = st.session_state.messages,
                    model_name   = selected_model,
                    placeholder  = placeholder,
                )

                if context_docs:
                    with st.expander("📎 Sources"):
                        for doc in context_docs:
                            pg      = doc.metadata.get("page", "?")
                            ocr_tag = " · OCR" if doc.metadata.get("ocr") else ""
                            st.markdown(
                                f'<span class="model-pill">Page {pg}{ocr_tag}</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(doc.page_content[:250] + "...")

                if reasoning_trace:
                    with st.expander("🧠 Reasoning Trace"):
                        st.code(reasoning_trace, language=None)

                st.session_state.messages.append({
                    "role":      "assistant",
                    "content":   answer,
                    "sources":   context_docs,
                    "reasoning": reasoning_trace,
                })
        else:
            st.info("⚡ Click **Initialize Knowledge Base** in the sidebar to begin.")


def render_index_tab():
    st.markdown("### 🌲 Document Index Tree")
    st.markdown(
        "When PageIndex is active, documents are indexed into a hierarchical tree. "
        "The LLM reads section summaries to **reason** about where the answer lives."
    )
    st.markdown("---")

    if st.session_state.page_tree and st.session_state.page_tree.built:
        tree = st.session_state.page_tree
        st.success(f"✅ {len(tree.tree)} sections indexed")
        for node in tree.tree:
            c1, c2 = st.columns([1, 10])
            with c1:
                st.markdown(
                    f'<div style="background:rgba(167,139,250,0.15);border-radius:50%;'
                    f'width:32px;height:32px;display:flex;align-items:center;'
                    f'justify-content:center;color:#A78BFA;font-weight:700;'
                    f'font-size:0.85rem">{node["id"]}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                with st.expander(
                    f"**{node['title']}** — Pages {node['start_page']}-{node['end_page']}"
                ):
                    st.markdown(f"**Summary:** {node['summary']}")
                    st.markdown("**Pages:** " + " ".join([f"`{p}`" for p in node["pages"]]))
    elif st.session_state.rag_mode == "faiss" and st.session_state.kb_initialized:
        st.info("FAISS mode is active — document tree is only built in PageIndex mode.")
    else:
        st.info("Upload PDFs and initialize the knowledge base to see the document tree.")


def render_tables_tab(uploaded_files):
    st.markdown("### 📊 Extracted Tables")
    found = False
    for f in uploaded_files:
        tables = extract_tables(f.getvalue(), f.name)
        if tables:
            found = True
            st.markdown(f"**{f.name}**")
            for t in tables:
                st.markdown(
                    f'<span class="model-pill">Page {t["page"]}</span>',
                    unsafe_allow_html=True,
                )
                st.dataframe(t["df"], use_container_width=True)
                st.download_button(
                    "⬇️ Export CSV",
                    t["df"].to_csv(index=False),
                    file_name=f"{f.name}_p{t['page']}.csv",
                    mime="text/csv",
                    key=f"csv_{f.name}_{t['page']}",
                )
                st.divider()
    if not found:
        st.info("No tables detected in the uploaded PDFs.")


def render_vision_tab(uploaded_files, available_models):
    st.markdown("### 🖼️ Images & AI Vision Analysis")
    vision_ok = has_vision_model(available_models)
    if not vision_ok:
        st.warning("No vision model detected. Run: `ollama pull llama3.2-vision`")

    found = False
    for f in uploaded_files:
        imgs = extract_images(f.getvalue())
        if imgs:
            found = True
            st.markdown(f"**{f.name}** — {len(imgs)} image(s)")
            for img_data in imgs:
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(
                        img_data["image"],
                        caption=f"Page {img_data['page']} · {img_data['size']}",
                        use_container_width=True,
                    )
                with c2:
                    if vision_ok:
                        q = st.text_input(
                            "Question for AI:",
                            value="Describe this. If it's a chart, explain the data and trends.",
                            key=f"vq_{f.name}_{img_data['page']}_{img_data['index']}",
                        )
                        if st.button(
                            "👁️ Analyze",
                            key=f"vb_{f.name}_{img_data['page']}_{img_data['index']}",
                        ):
                            with st.spinner("Analyzing with vision model..."):
                                result = analyze_image(img_data["image"], available_models, q)
                            st.markdown(f"**Analysis:**\n\n{result}")
                    else:
                        st.info("Install a vision model to analyze images.")
                st.divider()
    if not found:
        st.info("No images found in the uploaded PDFs.")


def render_charts_tab(uploaded_files):
    st.markdown("### 📈 Interactive Chart Builder")

    all_tables = []
    for f in uploaded_files:
        for t in extract_tables(f.getvalue(), f.name):
            all_tables.append({"label": f"{f.name} · Page {t['page']}", "df": t["df"]})

    if not all_tables:
        st.info("No tables found. Upload PDFs with tabular data to generate charts.")
        return

    sel = st.selectbox("Select table:", [t["label"] for t in all_tables])
    df  = next(t["df"] for t in all_tables if t["label"] == sel).copy()

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    st.dataframe(df, use_container_width=True)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if len(all_cols) < 2:
        st.info("At least 2 columns are required to build a chart.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        x = st.selectbox("X Axis", all_cols)
    with c2:
        y = st.selectbox("Y Axis", num_cols if num_cols else all_cols)
    with c3:
        ctype = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie", "Area"])

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
                paper_bgcolor="#080C14", plot_bgcolor="#0D1117",
                font=dict(family="Syne", color="#E2E8F0"),
                title_font=dict(size=16, color="#F1F5F9"),
                xaxis=dict(gridcolor="#1E293B", color="#64748B"),
                yaxis=dict(gridcolor="#1E293B", color="#64748B"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")
