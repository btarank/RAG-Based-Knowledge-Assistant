import logging
import time
import traceback
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank
from app.generation.query_rewriter import rewrite_query
from app.generation.context_builder import build_context, deduplicate_results
from app.generation.prompts import build_rag_prompt
from app.api.core.llm import get_llm_response, get_llm_streaming
from app.api.core.config import settings
from app.ingestion.embedder import get_collection_stats

logger = logging.getLogger("rag_assistant")
def answer_query(query: str, top_k: int = None, use_rewrite: bool = True, use_rerank: bool = True) -> dict:
    start_time = time.time()
    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        return {
        "query": query,
        "answer": "No documents have been uploaded yet. Please upload a document first.",
        "citations": [],
        "chunks_retrieved": 0,
        "total_time_seconds": round(time.time() - start_time, 2)
        }

    if top_k is None:
        top_k = settings.top_k

def answer_query(
    query: str,
    top_k: int = None,
    use_rewrite: bool = True,
    use_rerank: bool = True
) -> dict:
    """
    Full RAG pipeline (non-streaming):
    1. Check documents exist
    2. Rewrite query
    3. Hybrid search
    4. Deduplicate
    5. Re-rank
    6. Build context
    7. Generate grounded answer with citations
    """
    start_time = time.time()

    try:
        # Guard — no documents uploaded yet
        stats = get_collection_stats()
        if stats["total_chunks"] == 0:
            return {
                "query": query,
                "answer": "No documents have been uploaded yet. Please upload a document first.",
                "citations": [],
                "chunks_retrieved": 0,
                "reranked": False,
                "total_time_seconds": round(time.time() - start_time, 2)
            }

        if top_k is None:
            top_k = settings.top_k

        logger.info(f"Query received: {query[:60]}...")

        # Step 1: Query rewriting
        search_query = rewrite_query(query) if use_rewrite else query
        if search_query != query:
            logger.info(f"Query rewritten to: {search_query[:60]}...")

        # Step 2: Retrieve more candidates than needed for re-ranking
        retrieve_k = top_k * 3 if use_rerank else top_k
        results = hybrid_search(search_query, top_k=retrieve_k)
        logger.info(f"Retrieved {len(results)} chunks from hybrid search")

        if not results:
            return {
                "query": query,
                "rewritten_query": search_query if search_query != query else None,
                "answer": "I couldn't find any relevant information in the uploaded documents to answer this question.",
                "citations": [],
                "chunks_retrieved": 0,
                "reranked": False,
                "total_time_seconds": round(time.time() - start_time, 2)
            }

        # Step 3: Deduplicate
        results = deduplicate_results(results)

        # Step 4: Re-rank or just truncate
        if use_rerank:
            results = rerank(query, results, top_k=top_k)
            logger.info(f"Re-ranked down to {len(results)} chunks")
        else:
            results = results[:top_k]

        # Step 5: Build context and generate answer
        context, citation_map = build_context(results)
        messages = build_rag_prompt(query, context)
        answer = get_llm_response(messages, temperature=0.1)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Query completed in {elapsed}s — {len(results)} chunks used")

        return {
            "query": query,
            "rewritten_query": search_query if search_query != query else None,
            "answer": answer,
            "citations": citation_map,
            "chunks_retrieved": len(results),
            "reranked": use_rerank,
            "total_time_seconds": elapsed
        }

    except Exception as e:
        logger.error(f"Query failed: {e}\n{traceback.format_exc()}")
        raise


def answer_query_streaming(
    query: str,
    top_k: int = None,
    use_rewrite: bool = True,
    use_rerank: bool = True
):
    """
    Full RAG pipeline (streaming) — yields text tokens as they arrive.
    Citations are sent as a final JSON chunk after the answer text.
    """
    if top_k is None:
        top_k = settings.top_k

    try:
        stats = get_collection_stats()
        if stats["total_chunks"] == 0:
            yield "No documents have been uploaded yet. Please upload a document first."
            return

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

        # Send citations as a final parseable marker
        import json
        yield f"\n\n__CITATIONS__{json.dumps(citation_map)}"

    except Exception as e:
        logger.error(f"Streaming query failed: {e}\n{traceback.format_exc()}")
        yield f"\n\nError: {str(e)}"