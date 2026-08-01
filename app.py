import os
import datetime
import pandas as pd
import altair as alt
import streamlit as st
import chromadb

from ingest import main as run_ingest
from retrieve import dialectic_retrieve
from drift import drift_summary
from synthesize import synthesize
from diary_writer import save_diary_entry

st.set_page_config(page_title="Contradiction Detective", page_icon="🕵️", layout="centered")

password = st.text_input("Enter access password", type="password")
if password != st.secrets.get("APP_PASSWORD", ""):
    st.stop()

# ---------- Ingestion check (checks actual DATA count, not just folder existence) ----------
_client_check = chromadb.PersistentClient(path="./chroma_store")
_collection_check = _client_check.get_or_create_collection("notes")

if _collection_check.count() == 0:
    with st.spinner("First-time setup: embedding your diary... (this can take a minute)"):
        run_ingest()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Custom styling ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #161a23 100%);
    }
    .stApp, .stApp p, .stApp li, .stApp span,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    h1, h2, h3, h4, h5, h6 {
        color: #e6e6e6 !important;
    }
    .stApp .stMarkdown a {
        color: #ff8a5b !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161a23 0%, #0e1117 100%);
        border-right: 1px solid #2a2f3a;
    }
    [data-testid="stSidebar"] * {
        color: #e6e6e6 !important;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8a5b, #ff5b8f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9aa1ac;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 0.6rem;
    }
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #3a3f52;
        background-color: #1c202b;
        color: #e6e6e6;
        padding: 0.6rem;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #ff5b8f;
        color: #ff5b8f;
        background-color: #23273350;
        box-shadow: 0 0 12px #ff5b8f30;
    }
    .evidence-card-support {
        border-left: 4px solid #4ade80;
        background-color: #12241a;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .evidence-card-oppose {
        border-left: 4px solid #f87171;
        background-color: #241414;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .drift-banner {
        background-color: #1c1f2e;
        border: 1px solid #3a3f52;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<p class="main-title">🕵️ Contradiction Detective</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A diary that argues with your past self.</p>', unsafe_allow_html=True)
st.markdown(
    '<span style="background:#ff5b8f22; color:#ff5b8f; padding:4px 10px; '
    'border-radius:20px; font-size:0.8rem;">LOCAL · PRIVATE · FREE</span>',
    unsafe_allow_html=True,
)
st.write("")

# ---------- Timeline chart helper ----------
def render_timeline(supporting, opposing):
    rows = []
    for c in supporting:
        rows.append({"date": c["date"], "type": "Supporting"})
    for c in opposing:
        rows.append({"date": c["date"], "type": "Opposing"})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    chart = alt.Chart(df).mark_circle(size=180).encode(
        x=alt.X("date:T", title="Timeline"),
        y=alt.Y("type:N", title=""),
        color=alt.Color(
            "type:N",
            scale=alt.Scale(domain=["Supporting", "Opposing"], range=["#4ade80", "#f87171"]),
            legend=None,
        ),
        tooltip=["date:T", "type:N"],
    ).properties(height=150, background="transparent")
    st.altair_chart(chart, use_container_width=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📊 Diary Snapshot")
    try:
        count = _collection_check.count()
        st.metric("Entries embedded", count)
    except Exception:
        st.write("No data ingested yet.")

    if st.session_state.history:
        st.divider()
        st.subheader("🕓 Recent Questions")
        for h in st.session_state.history[:5]:
            st.caption(f"**{h['question']}**")
            st.caption(h["drift"])

    st.divider()
    st.caption("Built with a free, hosted dialectic RAG pipeline — Groq + ChromaDB + sentence-transformers.")

# ---------- Tabs ----------
tab_ask, tab_write = st.tabs(["💬 Ask a Question", "✍️ Write Today's Entry"])

with tab_write:
    st.write("Add today's entry — saved to your diary file **and** embedded immediately.")
    entry_date = st.date_input("Date", value=datetime.date.today())
    entry_text = st.text_area(
        "What happened today?", height=150, placeholder="Jogged in the morning, then..."
    )

    if st.button("💾 Save Entry", type="primary"):
        if entry_text.strip():
            with st.spinner("Saving and embedding your entry..."):
                save_diary_entry(entry_date, entry_text)
            st.success(f"Saved and embedded entry for {entry_date.strftime('%B %d, %Y')} ✅")
        else:
            st.warning("Write something before saving.")

with tab_ask:
    SAMPLE_QUESTIONS = [
        "Am I doing okay, or am I being too hard on myself?",
        "Has my confidence changed over time?",
        "Am I actually taking care of myself, or just saying I am?",
        "What tends to lift my mood on hard days?",
    ]
    st.write("**Try a sample question:**")
    cols = st.columns(2)
    picked = None
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True):
            picked = q

    query = st.text_input("Or ask your own question:", value=picked or "", placeholder="Type here...")

    if query:
        with st.spinner("🔎 Digging through supporting entries..."):
            result = dialectic_retrieve(query)
        with st.spinner("🧭 Hunting for the opposite point of view..."):
            drift_note = drift_summary(result["supporting"], result["opposing"])

        if not result["supporting"] and not result["opposing"]:
            st.warning(
                "⚠️ No matching diary entries were found for this question. "
                "The database may still be empty, or nothing in your diary relates closely "
                "enough to this question."
            )
        else:
            with st.spinner("🧩 Weighing both sides..."):
                answer = synthesize(query, result, drift_note)

            st.session_state.history.insert(0, {"question": query, "drift": drift_note})

            st.divider()
            st.subheader("💬 Answer")
            st.write(answer)

            st.markdown(
                f'<div class="drift-banner">📈 <b>Drift signal:</b> {drift_note}</div>',
                unsafe_allow_html=True,
            )
            render_timeline(result["supporting"], result["opposing"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✅ Supporting")
                if result["supporting"]:
                    for c in sorted(result["supporting"], key=lambda x: x["date"]):
                        st.markdown(
                            f'<div class="evidence-card-support"><b>{c["date"]}</b><br>{c["text"]}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No supporting evidence found.")
            with col2:
                st.markdown("#### ❌ Opposing")
                if result["opposing"]:
                    for c in sorted(result["opposing"], key=lambda x: x["date"]):
                        st.markdown(
                            f'<div class="evidence-card-oppose"><b>{c["date"]}</b><br>{c["text"]}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No opposing evidence found.")

            with st.expander("🔍 What the AI searched for"):
                st.write(f"**Your question:** {query}")
                st.write(f"**Opposite framing used for search:** {result['negated_query']}")
    else:
        st.write("👆 Pick a sample question above, or type your own, to get started.")