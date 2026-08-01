from llm_utils import chat


def synthesize(query: str, dialectic_result: dict, drift_note: str) -> str:
    supporting = dialectic_result["supporting"]
    opposing = dialectic_result["opposing"]

    # Guardrail: if there's truly no evidence, don't let the LLM invent
    # plausible-sounding "diary entries" that were never actually written.
    if not supporting and not opposing:
        return (
            "I couldn't find any diary entries relevant to this question. "
            "This usually means either the diary hasn't been fully embedded yet, "
            "or nothing you've written so far touches on this topic."
        )

    supporting_text = "\n".join(
        f"- ({c['date']}) {c['text']}" for c in supporting
    )
    opposing_text = "\n".join(
        f"- ({c['date']}) {c['text']}" for c in opposing
    )

    prompt = f"""You are analyzing a person's own diary to answer their question honestly,
including where their entries disagree with each other over time.

IMPORTANT: Only use the evidence provided below. Do not invent, paraphrase-as-fact,
or assume any diary entry, quote, or event that is not explicitly listed here.
If the evidence is thin, say so plainly instead of filling in gaps.

QUESTION: {query}

EVIDENCE SUPPORTING A YES / POSITIVE ANSWER:
{supporting_text if supporting_text else "(none found)"}

EVIDENCE SUGGESTING THE OPPOSITE:
{opposing_text if opposing_text else "(none found)"}

TIME-BASED SIGNAL: {drift_note}

Respond in exactly this structure, using ONLY the evidence above:
1. Direct answer (2 sentences max)
2. What the earlier entries suggest (quote or closely paraphrase only what's listed above)
3. What the later entries suggest (quote or closely paraphrase only what's listed above)
4. Whether the person's feelings seem to have changed, and why that matters
"""

    return chat(prompt)