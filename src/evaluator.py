
def evaluate_retrieval(retrieved: list[str], relevant: list[str]) -> float:
    """Calculate recall@K."""
    if not relevant:
        return 0.0
    hits = set(retrieved) & set(relevant)
    return len(hits) / len(relevant)

def evaluate_faithfulness(answer: str, context: list[str]) -> float:
    """Placeholder for faithfulness check (e.g., NLI or LLM-as-a-judge)."""
    # Simple keyword-based dummy implementation
    return 1.0 if any(c in answer for c in context) else 0.0
