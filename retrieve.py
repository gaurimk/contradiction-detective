import chromadb
from embed_utils import embed
from llm_utils import chat

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("notes")


def generate_negation_query(query: str) -> str:
    """Ask the LLM to produce the semantic opposite of the query."""
    prompt = (
        "Rewrite the following question so it asks for the OPPOSITE conclusion "
        "or an argument AGAINST the likely answer. Return only the rewritten "
        f"question, nothing else.\n\nQuestion: {query}"
    )

    return chat(prompt).strip()


def retrieve_chunks(query_text: str, n_results: int = 4):
    embedding = embed(query_text)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
    )

    return [
        {
            "text": doc,
            "date": meta["date"],
            "source": meta["source"],
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def dialectic_retrieve(query: str):
    supporting = retrieve_chunks(query)

    negated_query = generate_negation_query(query)

    opposing = retrieve_chunks(negated_query)

    return {
        "original_query": query,
        "negated_query": negated_query,
        "supporting": supporting,
        "opposing": opposing,
    }


if __name__ == "__main__":
    result = dialectic_retrieve(
        "Should we have used a microservices architecture?"
    )

    print("Negated query used:", result["negated_query"])

    print("\n--- SUPPORTING ---")
    for c in result["supporting"]:
        print(f"[{c['date']}] {c['text'][:120]}...")

    print("\n--- OPPOSING ---")
    for c in result["opposing"]:
        print(f"[{c['date']}] {c['text'][:120]}...")