from rank_bm25 import BM25Okapi
from app.ingestion.embedder import get_chroma_collection
from app.retrieval.semantic_search import SearchResult
import re

# Cache BM25 index in memory — rebuilt when documents change
_bm25_index = None
_bm25_chunk_ids = None
_bm25_chunk_texts = None
_bm25_chunk_metadata = None


def tokenize(text: str) -> list[str]:
    """Simple tokenizer — lowercase, split on non-alphanumeric."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens


def build_bm25_index():
    """
    Pull all chunks from ChromaDB and build a BM25 index.
    Call this after every document upload/delete.
    """
    global _bm25_index, _bm25_chunk_ids, _bm25_chunk_texts, _bm25_chunk_metadata

    collection = get_chroma_collection()

    # Get all documents from Chroma
    all_data = collection.get()

    if not all_data["ids"]:
        _bm25_index = None
        return

    _bm25_chunk_ids = all_data["ids"]
    _bm25_chunk_texts = all_data["documents"]
    _bm25_chunk_metadata = all_data["metadatas"]

    tokenized_corpus = [tokenize(text) for text in _bm25_chunk_texts]
    _bm25_index = BM25Okapi(tokenized_corpus)

    print(f"BM25 index built: {len(_bm25_chunk_ids)} chunks")


def keyword_search(query: str, top_k: int = 10) -> list[SearchResult]:
    """
    Search using BM25 keyword matching.
    Returns top_k chunks ranked by keyword relevance.
    """
    global _bm25_index

    # Build index on first use
    if _bm25_index is None:
        build_bm25_index()

    if _bm25_index is None:
        return []  # no documents yet

    tokenized_query = tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top_k indices sorted by score descending
    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    search_results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue  # skip zero-relevance results

        search_results.append(SearchResult(
            chunk_id=_bm25_chunk_ids[idx],
            text=_bm25_chunk_texts[idx],
            score=float(scores[idx]),
            metadata=_bm25_chunk_metadata[idx],
            source="keyword"
        ))

    return search_results