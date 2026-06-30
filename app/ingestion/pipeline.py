import shutil
from pathlib import Path
from app.ingestion.parser import parse_document
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_and_store, delete_document
from app.retrieval.keyword_search import build_bm25_index 
from app.api.core.config import settings
import time

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def ingest_document(file_path: str, filename: str) -> dict:
    """
    Full ingestion pipeline:
    1. Parse document → extract text per page
    2. Chunk pages → overlapping chunks with metadata
    3. Embed chunks → store in ChromaDB

    Returns a summary dict.
    """
    start_time = time.time()

    print(f"\n--- Ingesting: {filename} ---")

    # Step 1: Parse
    print("Step 1: Parsing document...")
    pages = parse_document(file_path)
    print(f"  Extracted {len(pages)} pages")

    if not pages:
        raise ValueError("No text could be extracted from this document.")

    # Step 2: Chunk
    print("Step 2: Chunking...")
    chunks = chunk_document(pages)
    print(f"  Created {len(chunks)} chunks")

    # Step 3: Embed + Store
    print("Step 3: Embedding and storing...")
    stored = embed_and_store(chunks)

    print("Step 4: Rebuilding BM25 index...")   # ADD THIS
    build_bm25_index()
    
    elapsed = round(time.time() - start_time, 2)

    return {
        "filename": filename,
        "pages_parsed": len(pages),
        "chunks_created": len(chunks),
        "chunks_stored": stored,
        "processing_time_seconds": elapsed,
        "status": "success"
    }


def save_upload(file_bytes: bytes, filename: str) -> str:
    """Save uploaded file to disk and return the path."""
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return str(file_path)