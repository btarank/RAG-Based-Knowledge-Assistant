from app.api.core.llm import get_llm_response

QUERY_REWRITE_PROMPT = """You are a query optimization assistant. Rewrite the user's question to be more specific and search-friendly for a document retrieval system.

Rules:
- Keep the same intent, do not change the meaning
- Expand abbreviations and pronouns into specific terms if context allows
- Output ONLY the rewritten query, nothing else
- If the query is already clear and specific, return it unchanged
- Do not answer the question, only rewrite it

User question: {query}

Rewritten query:"""


def rewrite_query(query: str) -> str:
    """
    Use the LLM to rewrite an ambiguous query into a clearer search query.
    Falls back to the original query if rewriting fails.
    """
    try:
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        rewritten = get_llm_response(
            [{"role": "user", "content": prompt}],
            temperature=0.0  # deterministic — we want consistency, not creativity
        )
        rewritten = rewritten.strip().strip('"')

        # Safety check — if rewrite is empty or absurdly long, use original
        if not rewritten or len(rewritten) > len(query) * 5:
            return query

        return rewritten

    except Exception as e:
        print(f"Query rewrite failed: {e}, using original query")
        return query