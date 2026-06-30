from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ingestion.pipeline import ingest_document, save_upload
from app.ingestion.embedder import get_collection_stats, delete_document
from app.api.core.config import settings

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file and ingest it into the RAG system.
    """
    # Validate file type
    allowed_types = [".pdf", ".txt"]
    filename = file.filename
    ext = "." + filename.split(".")[-1].lower()

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_types}"
        )

    # Validate file size
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.max_upload_size_mb}MB"
        )

    # Save to disk
    file_path = save_upload(file_bytes, filename)

    # Run ingestion pipeline
    try:
        result = ingest_document(file_path, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return result

@router.post("/upload-batch")
async def upload_multiple_documents(files: list[UploadFile] = File(...)):
    """
    Upload multiple documents at once — for building a real multi-document corpus.
    """
    results = []
    for file in files:
        filename = file.filename
        ext = "." + filename.split(".")[-1].lower()

        if ext not in [".pdf", ".txt"]:
            results.append({"filename": filename, "status": "skipped", "reason": f"unsupported type {ext}"})
            continue

        file_bytes = await file.read()
        size_mb = len(file_bytes) / (1024 * 1024)

        if size_mb > settings.max_upload_size_mb:
            results.append({"filename": filename, "status": "skipped", "reason": "file too large"})
            continue

        file_path = save_upload(file_bytes, filename)

        try:
            result = ingest_document(file_path, filename)
            results.append(result)
        except Exception as e:
            results.append({"filename": filename, "status": "failed", "reason": str(e)})

    return {"total_files": len(files), "results": results}


@router.get("/stats")
async def collection_stats():
    """Return stats about stored documents."""
    return get_collection_stats()

from app.retrieval.keyword_search import build_bm25_index 
@router.delete("/{filename}")
async def delete_document_endpoint(filename: str):
    """Delete all chunks for a specific document."""
    try:
        delete_document(filename)
        build_bm25_index() 
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#For viewing actual chunks 
@router.get("/debug/peek")
async def peek_chunks(limit: int = 10):
    """View raw stored chunks — for debugging only."""
    from app.ingestion.embedder import get_chroma_collection
    collection = get_chroma_collection()
    data = collection.get(limit=limit)

    return [
        {
            "id": data["ids"][i],
            "text_preview": data["documents"][i][:200],
            "metadata": data["metadatas"][i]
        }
        for i in range(len(data["ids"]))
    ]    