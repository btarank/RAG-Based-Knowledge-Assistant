from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank  # ADD THIS
from app.generation.query_rewriter import rewrite_query
from app.generation.context_builder import build_context, deduplicate_results
from app.generation.prompts import build_rag_prompt
from app.api.core.llm import get_llm_response, get_llm_streaming
from app.api.core.config import settings
import time


def answer_query(query: str, top_k: int = None, use_rewrite: bool = True, use_rerank: bool = True) -> dict:
    start_time = time.time()

    if top_k is None:
        top_k = settings.top_k

    search_query = rewrite_query(query) if use_rewrite else query

    # Retrieve more candidates than needed — re-ranker narrows it down
    retrieve_k = top_k * 3 if use_rerank else top_k
    results = hybrid_search(search_query, top_k=retrieve_k)

    if not results:
        return {
            "query": query,
            "answer": "I couldn't find any relevant information in the uploaded documents to answer this question.",
            "citations": [],
            "retrieval_time_seconds": round(time.time() - start_time, 2)
        }

    results = deduplicate_results(results)

    # Re-rank with cross-encoder
    if use_rerank:
        results = rerank(query, results, top_k=top_k)
    else:
        results = results[:top_k]

    context, citation_map = build_context(results)

    messages = build_rag_prompt(query, context)
    answer = get_llm_response(messages, temperature=0.1)

    elapsed = round(time.time() - start_time, 2)

    return {
        "query": query,
        "rewritten_query": search_query if search_query != query else None,
        "answer": answer,
        "citations": citation_map,
        "chunks_retrieved": len(results),
        "reranked": use_rerank,
        "total_time_seconds": elapsed
    }


def answer_query_streaming(query: str, top_k: int = None, use_rewrite: bool = True, use_rerank: bool = True):
    if top_k is None:
        top_k = settings.top_k

    search_query = rewrite_query(query) if use_rewrite else query

    retrieve_k = top_k * 3 if use_rerank else top_k
    results = hybrid_search(search_query, top_k=retrieve_k)

    if not results:
        yield "I couldn't find any relevant information in the uploaded documents to answer this question."
        return

    results = deduplicate_results(results)

    if use_rerank:
        results = rerank(query, results, top_k=top_k)
    else:
        results = results[:top_k]

    context, citation_map = build_context(results)
    messages = build_rag_prompt(query, context)

    for token in get_llm_streaming(messages, temperature=0.1):
        yield token

    import json
    yield f"\n\n__CITATIONS__{json.dumps(citation_map)}"