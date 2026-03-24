import streamlit as st
import time
import random
from src.ingestion import load_and_chunk_pdfs
from src.utils import save_uploaded_files, clear_upload_dir, get_pdf_paths
from src.embeddings import add_chunks_to_vectorstore, load_vectorstore, reset_vectorstore
from src.retriever import get_retriever, get_semantic_retriever, get_bm25_retriever, get_hybrid_retriever, get_mmr_retriever
from loguru import logger
from src.chain import build_conversational_chain, ask
from src.evaluate import evaluate_pipeline

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — Multi-PDF RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Fix: Force clear Streamlit's sidebar localStorage state ─────────────────
# No additional JS needed if CSS and Page Config are correctly set.
# Sidebar state can be managed by st.set_page_config(initial_sidebar_state="expanded")
# and ensuring the toggle remains visible via CSS.
 

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header {
    background-color: transparent !important;
    border: none !important;
    height: 3rem !important;
}
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    background-color: #6c5ce7 !important;
    border-radius: 8px !important;
    margin: 10px !important;
    color: white !important;
}
.block-container { padding-top: 0rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f11;
    border-right: 1px solid #1e1e24;
}
[data-testid="stSidebar"] * { color: #e8e6f0 !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Serif Display', serif;
    color: #ffffff !important;
}

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2rem;
}
.app-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #6c5ce7, #a29bfe);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1a1a2e;
    margin: 0; line-height: 1.1;
}
.app-subtitle {
    font-size: 0.85rem;
    color: #6c6c8a;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* ── Stat cards ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 2rem;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #ebebf5;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    transition: box-shadow 0.2s;
}
.stat-card:hover { box-shadow: 0 4px 24px rgba(108,92,231,0.10); }
.stat-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #9090aa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #1a1a2e;
    line-height: 1;
}
.stat-delta {
    font-size: 0.75rem;
    color: #00b894;
    margin-top: 4px;
}

/* ── Upload zone ── */
.upload-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6c6c8a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
    display: block;
}

/* ── Chat bubble ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    margin-bottom: 1.5rem;
}
.bubble-wrap {
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.bubble-wrap.user { flex-direction: row-reverse; }

.avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 600;
    flex-shrink: 0;
}
.avatar.ai { background: #6c5ce7; color: white; }
.avatar.user { background: #dfe6e9; color: #2d3436; }

.bubble {
    max-width: 80%;
    border-radius: 18px;
    padding: 0.85rem 1.1rem;
    font-size: 0.92rem;
    line-height: 1.6;
}
.bubble.ai {
    background: #f4f3ff;
    border: 1px solid #ebe8ff;
    color: #2d2b4e;
    border-top-left-radius: 4px;
}
.bubble.user {
    background: #6c5ce7;
    color: white;
    border-top-right-radius: 4px;
}

/* ── Citation badge ── */
.citation {
    display: inline-block;
    background: #ebe8ff;
    color: #6c5ce7;
    border-radius: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 1px 6px;
    margin-left: 4px;
    cursor: pointer;
}

/* ── Source card ── */
.source-card {
    background: #fafafe;
    border: 1px solid #ebebf5;
    border-left: 3px solid #6c5ce7;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    color: #4a4a6a;
}
.source-card strong {
    display: block;
    font-size: 0.78rem;
    color: #6c5ce7;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

/* ── Section heading ── */
.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #1a1a2e;
    margin-bottom: 1rem;
    border-bottom: 1px solid #ebebf5;
    padding-bottom: 0.5rem;
}

/* ── RAGAS metric bar ── */
.metric-bar-wrap {
    margin: 0.5rem 0;
}
.metric-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #6c6c8a;
    margin-bottom: 4px;
}
.metric-bar-label span:last-child {
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    color: #1a1a2e;
}
.metric-bar-track {
    height: 6px;
    background: #ebebf5;
    border-radius: 99px;
    overflow: hidden;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #6c5ce7, #a29bfe);
}

/* ── Chunk explorer ── */
.chunk-pill {
    display: inline-block;
    background: #f4f3ff;
    border: 1px solid #d6d0ff;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #6c5ce7;
    margin: 3px;
    cursor: pointer;
    font-family: 'DM Mono', monospace;
}
.chunk-pill:hover { background: #ebe8ff; }

/* ── Input ── */
.stTextInput input {
    border-radius: 12px !important;
    border: 1.5px solid #d6d0ff !important;
    padding: 0.6rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus {
    border-color: #6c5ce7 !important;
    box-shadow: 0 0 0 3px rgba(108,92,231,0.12) !important;
}

/* ── Button ── */
.stButton button {
    background: #6c5ce7 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.5rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: background 0.2s, transform 0.1s !important;
}
.stButton button:hover {
    background: #5a4bd1 !important;
    transform: translateY(-1px) !important;
}
.stButton button:active { transform: translateY(0) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #d6d0ff !important;
    border-radius: 16px !important;
    background: #fafafe !important;
    padding: 1rem !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #6c5ce7, #a29bfe) !important;
    border-radius: 99px !important;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #ebebf5;
    margin: 1.5rem 0;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid #ebebf5;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #9090aa !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    color: #6c5ce7 !important;
    background: #f4f3ff !important;
    border-bottom: 2px solid #6c5ce7 !important;
}

</style>
""", unsafe_allow_html=True)


# ─── Session State Init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdfs_loaded" not in st.session_state:
    st.session_state.pdfs_loaded = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "ragas_scores" not in st.session_state:
    st.session_state.ragas_scores = {}


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("<p style='color:#9090aa;font-size:0.8rem;margin-top:-10px;'>Multi-PDF RAG Assistant</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📂 Load Documents")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.session_state.pdfs_loaded = [f.name for f in uploaded_files]
        for f in uploaded_files:
            st.markdown(f"<div style='font-size:0.78rem;color:#a29bfe;padding:3px 0;'>📄 {f.name}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Settings")

    chunk_size = st.slider("Chunk Size (tokens)", 128, 1024, 512, 64)
    chunk_overlap = st.slider("Chunk Overlap", 0, 256, 64, 16)

    embedding_model = st.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "text-embedding-3-small (OpenAI)", "BAAI/bge-small-en"]
    )

    retrieval_mode = st.radio(
        "Retrieval Mode",
        ["Semantic Search", "Hybrid (BM25 + Semantic)", "MMR Reranking"],
        index=1
    )

    # Map human-readable radio label -> retriever.py mode string
    RETRIEVAL_MODE_MAP = {
        "Semantic Search":          "semantic",
        "Hybrid (BM25 + Semantic)": "hybrid",
        "MMR Reranking":            "mmr",
    }
    selected_mode = RETRIEVAL_MODE_MAP[retrieval_mode]

    top_k = st.slider("Top-K Chunks Retrieved", 1, 10, 4)

    st.markdown("---")

    if st.button("Ingest & Index PDFs"):
        if uploaded_files:
            with st.spinner("Processing documents..."):

                # Step 1 — save to disk
                pdf_paths = save_uploaded_files(uploaded_files)

                # Step 2 — chunk
                chunks = load_and_chunk_pdfs(
                    pdf_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                # Step 3 — embed + store (returns the vectorstore directly)
                vectorstore = add_chunks_to_vectorstore(chunks)

                # Step 4 — build retriever
                retriever = get_retriever(
                    vectorstore,
                    chunks=chunks,          # ← BM25 needs raw chunks
                    mode=selected_mode,
                    k=top_k
                )

                # Step 5 — save everything to session_state
                st.session_state.chunks      = chunks        # ← was missing!
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever   = retriever     # ← was missing!
                st.session_state.chunk_count = len(chunks)
                st.session_state.chain       = build_conversational_chain(retriever)

            st.session_state.ingested    = True

        st.success(f"Indexed {len(uploaded_files)} PDF(s) → {len(chunks)} chunks")

    else:
        st.warning("Please upload at least one PDF first.")

    st.markdown("---")
    st.markdown("<p style='font-size:0.72rem;color:#555566;'>Built with LangChain · ChromaDB · RAGAS</p>", unsafe_allow_html=True)


# ─── Main Area ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">🧠</div>
  <div>
    <p class="app-title">DocMind</p>
    <p class="app-subtitle">Ask questions across multiple PDFs with cited, grounded answers</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Stat cards
pdf_count = len(st.session_state.pdfs_loaded)
chunk_display = st.session_state.chunk_count if st.session_state.ingested else 0
msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-label">PDFs Loaded</div>
    <div class="stat-value">{pdf_count}</div>
    <div class="stat-delta">{'ready to query' if st.session_state.ingested else 'not indexed yet'}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Chunks Indexed</div>
    <div class="stat-value">{chunk_display}</div>
    <div class="stat-delta">chunk size: {chunk_size} tokens</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Queries Asked</div>
    <div class="stat-value">{msg_count}</div>
    <div class="stat-delta">top-k: {top_k} · {retrieval_mode.split()[0]}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬  Chat", "📊  RAGAS Evaluation", "🔍  Chunk Explorer"])


# ══ Tab 1: Chat ══════════════════════════════════════════════════════════════
with tab1:

    # Render chat history
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center;padding:3rem 1rem;color:#9090aa;'>
          <div style='font-size:2.5rem;margin-bottom:12px;'>💬</div>
          <p style='font-size:1rem;font-weight:500;color:#6c6c8a;'>No conversation yet</p>
          <p style='font-size:0.82rem;'>Upload PDFs → Ingest → Ask anything</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="bubble-wrap user">
                  <div class="avatar user">You</div>
                  <div class="bubble user">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                sources_html = ""
                if "sources" in msg and msg["sources"]:
                    src = msg["sources"][0]   # ← only take the first source
                    sources_html += f'<div class="source-card"><strong>📄 {src["file"]} · p.{src["page"]}</strong>{src["snippet"]}</div>'
                    
                st.markdown(f"""
                <div class="bubble-wrap ai">
                  <div class="avatar ai">AI</div>
                  <div>
                    <div class="bubble ai">{msg["content"]}</div>
                    {sources_html}
                  </div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Input row
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            "Ask a question",
            placeholder="e.g. What are the key findings in chapter 3?",
            label_visibility="collapsed",
            key="query_input"
        )
    with col_btn:
        send = st.button("Send →", use_container_width=True)

    # Handle send
    # ─── Chat send handler ────────────────────────────────────────────────────────
    if send and user_query:
        if not st.session_state.ingested:
            st.warning("⚠️ Please upload and index your PDFs first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})  # ← uncommented

        # Build chat_history from confirmed user+assistant pairs only
        history = st.session_state.messages
        chat_history = [
            (history[i]["content"], history[i+1]["content"])
            for i in range(0, len(history) - 1, 2)
            if history[i]["role"] == "user" and history[i+1]["role"] == "assistant"
        ]

        with st.spinner("Retrieving & generating answer..."):
            response = ask(                          # ← real call inside spinner
                st.session_state.chain,
                user_query,
                chat_history=chat_history
            )

        st.session_state.messages.append({
            "role":    "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑 Clear conversation"):
            st.session_state.messages = []
            reset_vectorstore()
            clear_upload_dir()
            st.rerun()


# ══ Tab 2: RAGAS Evaluation ═══════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-heading">RAGAS Pipeline Evaluation</p>', unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:0.88rem;color:#6c6c8a;margin-bottom:1.5rem;'>
    RAGAS evaluates your RAG pipeline without human labels. Add a test dataset (question + ground truth answer pairs) and run evaluation below.
    </p>
    """, unsafe_allow_html=True)

    col_eval1, col_eval2 = st.columns([2, 1])

    with col_eval1:
        eval_file = st.file_uploader("Upload eval dataset (JSON)", type=["json"], key="eval_upload")
        # num_questions = st.slider("Or generate synthetic test questions", 5, 50, 15, 5)
        num_questions = st.slider(
            "Number of synthetic test questions",
            min_value=3,
            max_value=5,      # ← was 50, way too high for free tier
            value=5,
            step=1,
            help="Keep at 5 or below on Groq free tier. Each question = 4 LLM calls."
)

    with col_eval2:
        st.markdown("""
        <div style='background:#fafafe;border:1px solid #ebebf5;border-radius:12px;padding:1rem;font-size:0.8rem;color:#6c6c8a;'>
        <strong style='color:#1a1a2e;'>Dataset format:</strong><br><br>
        <code style='font-size:0.75rem;'>[{"question": "...", "ground_truth": "..."}]</code>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Run RAGAS Evaluation"):
        if not st.session_state.ingested:
            st.warning("Index your PDFs first.")
        else:
            with st.spinner("Running RAGAS evaluation..."):

                # Mode B — if user uploaded a JSON file
                json_path = None
                if eval_file:
                    json_path = "eval_dataset.json"
                    with open(json_path, "wb") as f:
                        f.write(eval_file.read())

            scores = evaluate_pipeline(
                chain=st.session_state.chain,
                chunks=st.session_state.chunks,
                json_path=json_path,           # None = synthetic mode
                num_synthetic=num_questions,   # from sidebar slider
            )

        st.session_state.ragas_scores = scores

    if st.session_state.ragas_scores:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<p class="section-heading">Evaluation Results</p>', unsafe_allow_html=True)

        scores = st.session_state.ragas_scores
        avg_score = round(sum(scores.values()) / len(scores), 2)

        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        cols = [col_a, col_b, col_c, col_d, col_e]
        metric_list = list(scores.items())

        with cols[0]:
            st.markdown(f"""
            <div class="stat-card" style='text-align:center;'>
              <div class="stat-label">Avg Score</div>
              <div class="stat-value" style='color:#6c5ce7;'>{avg_score}</div>
              <div class="stat-delta">overall</div>
            </div>""", unsafe_allow_html=True)

        for idx, (metric, score) in enumerate(metric_list):
            color = "#00b894" if score >= 0.85 else "#fdcb6e" if score >= 0.70 else "#d63031"
            with cols[idx + 1]:
                st.markdown(f"""
                <div class="stat-card" style='text-align:center;'>
                  <div class="stat-label">{metric}</div>
                  <div class="stat-value" style='color:{color};font-size:1.5rem;'>{score}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        for metric, score in scores.items():
            pct = int(score * 100)
            st.markdown(f"""
            <div class="metric-bar-wrap">
              <div class="metric-bar-label">
                <span>{metric}</span>
                <span>{score}</span>
              </div>
              <div class="metric-bar-track">
                <div class="metric-bar-fill" style="width:{pct}%;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Tip for your resume:** Screenshot these scores and include them in your README. Mention the metrics in interviews — very few candidates do this!")


# ══ Tab 3: Chunk Explorer ═════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-heading">Chunk Explorer</p>', unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:0.88rem;color:#6c6c8a;margin-bottom:1.5rem;'>
    Visualize how your documents are split into chunks. Understanding chunking is crucial — it directly affects retrieval quality.
    </p>
    """, unsafe_allow_html=True)

    if not st.session_state.ingested:
        st.info("Index your PDFs first to explore chunks.")
    else:
        filter_pdf = st.selectbox("Filter by PDF", ["All"] + st.session_state.pdfs_loaded)
        search_chunk = st.text_input("Search within chunks", placeholder="keyword...")

        # Mock chunks
        mock_chunks = [
            {"id": f"chunk_{i}", "tokens": random.randint(80, chunk_size), "page": random.randint(1, 15), "preview": f"...sample text from chunk {i} — this is what gets embedded and stored in ChromaDB for retrieval..."}
            for i in range(1, 13)
        ]

        st.markdown("**Chunks (click to expand):**")
        pills_html = "".join([f'<span class="chunk-pill">chunk_{i} · p.{c["page"]}</span>' for i, c in enumerate(mock_chunks, 1)])
        st.markdown(f'<div style="margin-bottom:1rem;">{pills_html}</div>', unsafe_allow_html=True)

        for i, chunk in enumerate(mock_chunks[:6], 1):
            with st.expander(f"chunk_{i}  ·  page {chunk['page']}  ·  {chunk['tokens']} tokens"):
                st.markdown(f"""
                <div class="source-card">
                  <strong>chunk_{i} · {chunk['tokens']} tokens</strong>
                  {chunk['preview']}
                </div>
                """, unsafe_allow_html=True)