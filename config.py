import streamlit as st


def set_page_config():
    st.set_page_config(
        page_title="Private AI v4.0",
        page_icon="🛡️",
        layout="wide"
    )


def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    background-color: #080C14 !important;
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
}
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
.main .block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

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
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.2);
    color: #60A5FA;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-right: 0.5rem;
    margin-bottom: 1.5rem;
}
.pi-badge {
    background: rgba(167,139,250,0.1) !important;
    border-color: rgba(167,139,250,0.2) !important;
    color: #A78BFA !important;
}
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
.sidebar-logo { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 0.2rem; }
.sidebar-version { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #475569; margin-bottom: 1.5rem; }
.status-ok {
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.2);
    color: #4ADE80; padding: 0.4rem 0.8rem; border-radius: 6px;
    font-size: 0.78rem; font-family: 'Space Mono', monospace;
    margin-bottom: 0.4rem; display: block;
}
.status-warn {
    background: rgba(234,179,8,0.1); border: 1px solid rgba(234,179,8,0.2);
    color: #FDE047; padding: 0.4rem 0.8rem; border-radius: 6px;
    font-size: 0.78rem; font-family: 'Space Mono', monospace;
    margin-bottom: 0.4rem; display: block;
}
[data-testid="stSelectbox"] > div > div {
    background: #0D1117 !important; border: 1px solid #1E293B !important;
    border-radius: 8px !important; color: #E2E8F0 !important;
}
[data-testid="stFileUploader"] {
    background: #0D1117 !important; border: 1px dashed #1E293B !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1E40AF, #7C3AED) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
    font-size: 0.85rem !important; padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.3) !important;
}
[data-testid="stChatMessage"] {
    background: #0D1117 !important; border: 1px solid #1E293B !important;
    border-radius: 12px !important; margin-bottom: 0.8rem !important; padding: 1rem !important;
}
[data-testid="stChatInput"] { background: #0D1117 !important; border: 1px solid #1E293B !important; border-radius: 10px !important; }
[data-testid="stChatInput"] textarea { background: #0D1117 !important; color: #E2E8F0 !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stTabs"] [role="tablist"] { background: #0D1117 !important; border-bottom: 1px solid #1E293B !important; gap: 0.5rem !important; }
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important; color: #64748B !important;
    border: 1px solid transparent !important; border-radius: 8px 8px 0 0 !important;
    font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
    font-size: 0.85rem !important; padding: 0.6rem 1.2rem !important; transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(30,64,175,0.2), rgba(124,58,237,0.2)) !important;
    color: #60A5FA !important; border-color: #1E293B !important;
}
[data-testid="stDataFrame"] { border: 1px solid #1E293B !important; border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stMetric"] { background: #0D1117 !important; border: 1px solid #1E293B !important; border-radius: 10px !important; padding: 1rem !important; }
[data-testid="stMetricLabel"] { color: #64748B !important; font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #E2E8F0 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #4ADE80 !important; font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important; }
[data-testid="stExpander"] { background: #0D1117 !important; border: 1px solid #1E293B !important; border-radius: 8px !important; }
[data-testid="stAlert"] { border-radius: 8px !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #1E40AF, #7C3AED) !important; }
hr { border-color: #1E293B !important; }
.feature-card { background: #0D1117; border: 1px solid #1E293B; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; transition: border-color 0.2s; }
.feature-card:hover { border-color: #334155; }
.feature-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
.feature-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.9rem; color: #F1F5F9; margin-bottom: 0.3rem; }
.feature-desc { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #64748B; line-height: 1.5; }
.model-pill {
    display: inline-block; background: rgba(96,165,250,0.08);
    border: 1px solid rgba(96,165,250,0.15); color: #93C5FD;
    padding: 0.15rem 0.6rem; border-radius: 20px;
    font-size: 0.72rem; font-family: 'Space Mono', monospace;
}
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)
