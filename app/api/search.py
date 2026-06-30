from fastapi import APIRouter
from pydantic import BaseModel
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.semantic_search import semantic_search
from app.retrieval.keyword_search import keyword_search

router = APIRouter(prefix="/search", tags=["Search"])


class SearchQuery(BaseModel):
    query: str
    top_k: int = 5


@router.post("/hybrid")
async def search_hybrid(payload: SearchQuery):
    """Hybrid search — semantic + keyword fused with RRF."""
    results = hybrid_search(payload.query, top_k=payload.top_k)
    return {
        "query": payload.query,
        "method": "hybrid",
        "results": [
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:300],  # preview only
                "score": round(r.score, 4),
                "source_file": r.metadata.get("source_file"),
                "page_number": r.metadata.get("page_number"),
            }
            for r in results
        ]
    }


@router.post("/semantic")
async def search_semantic(payload: SearchQuery):
    """Pure semantic search — for comparison/debugging."""
    results = semantic_search(payload.query, top_k=payload.top_k)
    return {
        "query": payload.query,
        "method": "semantic",
        "results": [
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:300],
                "score": round(r.score, 4),
                "source_file": r.metadata.get("source_file"),
                "page_number": r.metadata.get("page_number"),
            }
            for r in results
        ]
    }


@router.post("/keyword")
async def search_keyword(payload: SearchQuery):
    """Pure BM25 keyword search — for comparison/debugging."""
    results = keyword_search(payload.query, top_k=payload.top_k)
    return {
        "query": payload.query,
        "method": "keyword",
        "results": [
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:300],
                "score": round(r.score, 4),
                "source_file": r.metadata.get("source_file"),
                "page_number": r.metadata.get("page_number"),
            }
            for r in results
        ]
    }