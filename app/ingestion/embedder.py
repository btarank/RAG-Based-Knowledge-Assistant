import chromadb
from chromadb.config import Settings as ChromaSettings
from app.api.core.embeddings import embed_texts
from app.api.core.config import settings
from app.ingestion.chunker import Chunk
from pathlib import Path
import os

# Global chroma client
_chroma_client = None
_collection = None


def get_chroma_collection():
    """Get or create the ChromaDB collection."""
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    # Ensure storage directory exists
    Path(settings.vector_db_path).mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(
        path=settings.vector_db_path,
        settings=ChromaSettings(anonymized_telemetry=False)
    )

    _collection = _chroma_client.get_or_create_collection(
        name="rag_documents",
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )

    return _collection


def embed_and_store(chunks: list[Chunk], batch_size: int = 32) -> int:
    """
    Generate embeddings for all chunks and store in ChromaDB.
    Returns number of chunks stored.
    """
    collection = get_chroma_collection()

    # Process in batches to avoid memory issues
    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [c.metadata for c in batch]

        # Generate embeddings
        print(f"Embedding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        embeddings = embed_texts(texts)

        # Store in ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

        total_stored += len(batch)

    print(f"Stored {total_stored} chunks in ChromaDB")
    return total_stored


def delete_document(source_file: str):
    """Delete all chunks belonging to a specific document."""
    collection = get_chroma_collection()

    results = collection.get(
        where={"source_file": source_file}
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"Deleted {len(results['ids'])} chunks for {source_file}")


def get_collection_stats() -> dict:
    """Return stats about what's stored in ChromaDB."""
    collection = get_chroma_collection()
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": collection.name,
        "vector_db_path": settings.vector_db_path
    }