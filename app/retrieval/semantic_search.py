from app.ingestion.embedder import get_chroma_collection
from app.api.core.embeddings import embed_query
from dataclasses import dataclass

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict
    source: str  # "semantic" or "keyword"


def semantic_search(query: str, top_k: int = 10) -> list[SearchResult]:
    """
    Search ChromaDB using vector similarity (cosine distance).
    Returns top_k most semantically similar chunks.
    """
    collection = get_chroma_collection()

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    search_results = []

    if not results["ids"][0]:
        return search_results

    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # Chroma returns cosine distance (0 = identical, 2 = opposite)
        # Convert to similarity score (1 = identical, 0 = opposite)
        similarity = 1 - (distance / 2)

        search_results.append(SearchResult(
            chunk_id=chunk_id,
            text=text,
            score=similarity,
            metadata=metadata,
            source="semantic"
        ))

    return search_results