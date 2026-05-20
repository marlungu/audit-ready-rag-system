"""
Streamlit demo UI for the Audit-Ready RAG System.

Run the FastAPI server first (make serve), then:
    streamlit run demo_ui.py
"""

import os
import streamlit as st
import requests

API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="USCIS Policy Assistant",
    page_icon="🏛️",
    layout="wide",
)

# ── Custom styling ───────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container { max-width: 900px; }
    .confidence-high { color: #0f5132; background: #d1e7dd; padding: 4px 12px;
        border-radius: 4px; font-weight: 600; }
    .confidence-low { color: #664d03; background: #fff3cd; padding: 4px 12px;
        border-radius: 4px; font-weight: 600; }
    .confidence-insufficient { color: #842029; background: #f8d7da;
        padding: 4px 12px; border-radius: 4px; font-weight: 600; }
    .source-card { background: #f8f9fa; padding: 12px 16px; border-radius: 6px;
        border-left: 3px solid #1a2b4a; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────
st.title("USCIS Policy Assistant")
st.caption("Audit-Ready RAG System — Archetype Core")


# ── Sidebar callbacks ────────────────────────────────────────
def set_question(q):
    st.session_state["question_field"] = q
    # st.session_state["auto_submit"] = True


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=20, value=5)

    st.markdown("---")
    st.markdown("### Sample Questions")

    sample_questions = [
        "What are the eligibility requirements for naturalization?",
        "What is the continuous residence requirement for citizenship?",
        "What are the grounds for inadmissibility?",
        "Can I travel while my green card application is pending?",
        "How long do I have to live in the US to become a citizen?",
        "What is the best pizza in New York?",
    ]

    for q in sample_questions:
        st.button(q, key=q, use_container_width=True, on_click=set_question, args=(q,))

    st.markdown("---")
    st.markdown("### System Health")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        db = health.get("database", {})
        st.markdown(f"**Status:** {health.get('status', 'unknown')}")
        st.markdown(f"**Database:** {db.get('status', 'unknown')}")
        st.markdown(f"**Chunks indexed:** {db.get('indexed_chunks', 'N/A'):,}")
        st.markdown(f"**pgvector:** {'enabled' if db.get('pgvector_enabled') else 'N/A'}")
    except Exception:
        st.error("API not reachable. Is the server running?")

# ── Main input ───────────────────────────────────────────────
question = st.text_input(
    "Ask a question about USCIS immigration policy:",
    value=st.session_state.get("question_input", ""),
    key="question_field",
    placeholder="e.g., What are the eligibility requirements for naturalization?",
)

col_btn, col_spacer = st.columns([1, 4])
with col_btn:
    submitted = st.button("Ask", type="primary", use_container_width=True)

# Auto-submit when a sidebar button was clicked
# if st.session_state.get("auto_submit"):
#     submitted = True
#     st.session_state["auto_submit"] = False

# ── Query + Display ──────────────────────────────────────────
if submitted and question:
    with st.spinner("Searching policy documents..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Run `make serve` first.")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    # ── Confidence badge ─────────────────────────────────────
    confidence = data["confidence"]
    badge_class = f"confidence-{confidence}"
    st.markdown(
        f'<span class="{badge_class}">{confidence.upper()}</span>',
        unsafe_allow_html=True,
    )

    # ── Metrics row ──────────────────────────────────────────
    retrieval = data["retrieval"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", confidence.upper())
    col2.metric(
        "Top Similarity",
        f"{retrieval['top_similarity']:.4f}" if retrieval["top_similarity"] else "N/A",
    )
    col3.metric("Chunks Retrieved", retrieval["total_chunks_retrieved"])
    col4.metric("Chunks Sent to LLM", retrieval["chunks_sent_to_model"])

    # ── Answer ───────────────────────────────────────────────
    st.markdown("### Answer")
    st.markdown(data["answer"])

    # ── Sources ──────────────────────────────────────────────
    if data["sources"]:
        st.markdown("### Sources")
        for s in data["sources"]:
            st.markdown(
                f'<div class="source-card">'
                f"<strong>{s['document_title']}</strong>, "
                f"PDF Page {s['page_number']} &nbsp;·&nbsp; "
                f"Chunk {s['chunk_index']} &nbsp;·&nbsp; "
                f"Similarity: {s['similarity']:.4f} &nbsp;·&nbsp; "
                f"Query: <em>{s['matched_query']}</em>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Expanded Queries ─────────────────────────────────────
    if retrieval["expanded_queries"]:
        st.markdown("### Expanded Queries")
        for i, q in enumerate(retrieval["expanded_queries"], 1):
            st.markdown(f"{i}. {q}")

    # ── Audit Trail ──────────────────────────────────────────
    st.markdown("### Audit Trail")
    audit = data["audit"]
    audit_col1, audit_col2 = st.columns(2)
    with audit_col1:
        st.markdown(f"**Query ID:** `{audit['query_id']}`")
        st.markdown(f"**Model:** `{audit['model_id']}`")
        st.markdown(f"**Embedding Model:** `{audit['embedding_model_id']}`")
    with audit_col2:
        st.markdown(f"**Temperature:** `{audit['temperature']}`")
        st.markdown(f"**Timestamp:** `{audit['timestamp']}`")

    # ── Raw JSON (collapsible) ───────────────────────────────
    with st.expander("View raw JSON response"):
        st.json(data)