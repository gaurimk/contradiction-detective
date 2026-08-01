from llm_utils import chat


def synthesize(query: str, dialectic_result: dict, drift_note: str) -> str:
    supporting_text = "\n".join(
        f"- ({c['date']}) {c['text']}"
        for c in dialectic_result["supporting"]
    )

    opposing_text = "\n".join(
        f"- ({c['date']}) {c['text']}"
        for c in dialectic_result["opposing"]
    )

    prompt = f"""You are analyzing a person's own diary to answer their question honestly,
including where their entries disagree with each other over time.

QUESTION: {query}

EVIDENCE SUPPORTING A YES / POSITIVE ANSWER:
{supporting_text if supporting_text else "(none found)"}

EVIDENCE SUGGESTING THE OPPOSITE:
{opposing_text if opposing_text else "(none found)"}

TIME-BASED SIGNAL: {drift_note}

Respond in exactly this structure:
1. Direct answer (2 sentences max)
2. What the earlier entries suggest
3. What the later entries suggest
4. Whether the person's feelings seem to have changed, and why that matters.
"""

    return chat(prompt)