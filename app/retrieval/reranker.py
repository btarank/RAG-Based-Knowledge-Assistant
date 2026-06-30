from sentence_transformers import CrossEncoder
from app.retrieval.semantic_search import SearchResult

# Loaded once, reused across requests
_cross_encoder = None

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"Loading re-ranker model: {RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
        print("Re-ranker model loaded.")
    return _cross_encoder


def rerank(query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
    """
    Re-score hybrid search results using a cross-encoder.
    Returns the top_k results sorted by the new, more accurate scores.
    """
    if not results:
        return []

    model = get_reranker()

    # Cross-encoder expects pairs: (query, chunk_text)
    pairs = [(query, r.text) for r in results]
    scores = model.predict(pairs)

    # Attach new scores and sort descending
    reranked = []
    for result, score in zip(results, scores):
        reranked.append(SearchResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=float(score),
            metadata=result.metadata,
            source="reranked"
        ))

    reranked.sort(key=lambda r: r.score, reverse=True)

    return reranked[:top_k]