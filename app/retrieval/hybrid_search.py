from app.retrieval.semantic_search import semantic_search, SearchResult
from app.retrieval.keyword_search import keyword_search
from app.api.core.config import settings


def reciprocal_rank_fusion(
    semantic_results: list[SearchResult],
    keyword_results: list[SearchResult],
    k: int = 60
) -> list[SearchResult]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.
    RRF score = sum of 1/(k + rank) across all lists a chunk appears in.

    k=60 is the standard constant from the original RRF paper —
    it dampens the impact of very high ranks so one list doesn't dominate.
    """
    rrf_scores = {}
    chunk_lookup = {}

    # Score from semantic results
    for rank, result in enumerate(semantic_results):
        chunk_id = result.chunk_id
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        chunk_lookup[chunk_id] = result

    # Score from keyword results
    for rank, result in enumerate(keyword_results):
        chunk_id = result.chunk_id
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        if chunk_id not in chunk_lookup:
            chunk_lookup[chunk_id] = result

    # Sort by fused RRF score, descending
    sorted_chunk_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    fused_results = []
    for chunk_id in sorted_chunk_ids:
        original = chunk_lookup[chunk_id]
        fused_results.append(SearchResult(
            chunk_id=original.chunk_id,
            text=original.text,
            score=rrf_scores[chunk_id],
            metadata=original.metadata,
            source="hybrid"
        ))

    return fused_results


def hybrid_search(query: str, top_k: int = None) -> list[SearchResult]:
    """
    Full hybrid search pipeline:
    1. Run semantic search
    2. Run keyword (BM25) search
    3. Fuse both rankings with RRF
    4. Return top_k fused results
    """
    if top_k is None:
        top_k = settings.top_k

    # Retrieve more candidates than needed from each method (helps fusion quality)
    candidate_k = max(top_k * 2, 10)

    semantic_results = semantic_search(query, top_k=candidate_k)
    keyword_results = keyword_search(query, top_k=candidate_k)

    fused = reciprocal_rank_fusion(semantic_results, keyword_results)

    return fused[:top_k]