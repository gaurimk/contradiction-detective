# 🕵️ Contradiction Detective

A tool that reads through diary/journal entries and surfaces where you've contradicted yourself over time — plus whether your views on something have genuinely *drifted* rather than just been inconsistent day-to-day.

**🔗 Live demo:** https://contradiction-detective-tefhbx7szjrpunrsgnnh3v.streamlit.app/

## The problem

If you wrote in a diary every day for a year, your feelings about things would naturally change — confident in January, low in June, okay again by October. But if you asked a friend "was I confident about myself this year?", they'd likely just recall whatever they read most recently, or whatever matched the question. They wouldn't necessarily catch that your answer would've been completely different in June versus October.

Most AI diary-reading tools have the same blind spot: they answer a question by retrieving the most *similar*-sounding text, which tends to surface one dominant narrative and miss the contradictions entirely.

## What this does differently

Instead of retrieving the single closest match, the app deliberately searches for **both sides** of a question:
- one retrieval pass looks for entries that **support** the claim
- a second pass specifically looks for entries that **oppose** it

It then compares the two sets and explains where — and more importantly *when* — your view shifted. Because every entry is timestamped, the tool doesn't just say "you contradicted yourself," it says *"you felt this way in June, and the opposite by October"* — which is far more useful for understanding actual change versus noise.

In one sentence: it's a diary-reading assistant whose job is to catch you contradicting your past self and tell you exactly when that change happened, instead of answering like a normal Q&A chatbot.

## How it works
1. **Ingest** — diary entries are chunked and embedded (`sentence-transformers`) into a local vector store (`chromadb`).
2. **Dialectic retrieval** — for a given question or claim, the app retrieves both *supporting* and *opposing* evidence from past entries, not just the closest match.
3. **Synthesis** — an LLM (via Groq) weighs the two evidence sets and explains the contradiction, or confirms there isn't one.
4. **Drift detection** — evidence is bucketed by time period to distinguish "you flip-flopped in the same week" from "your view genuinely evolved over months."

## Stack
`Streamlit` · `ChromaDB` · `sentence-transformers` · `Groq` (LLM) · `pandas` / `altair` for visualization

## Running locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
You'll need a `GROQ_API_KEY` set in `.streamlit/secrets.toml` (see `.gitignore` — this file is never committed).

## Note on data
The included diary entries under `data/notes/` are **synthetic** (LLM-generated), written to demonstrate the contradiction/drift detection features. The app works with any plain-text diary in the same format.
