from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.core.config import settings
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.api.query import router as query_router
from app.api.evaluation import router as evaluation_router
app = FastAPI(
    title="RAG Assistant",
    description="Production-grade RAG system — free deployable stack",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(query_router)
app.include_router(evaluation_router)  
@app.get("/")
async def root():
    return {
        "status": "running",
        "env": settings.app_env,
        "llm": f"{settings.llm_provider} / {settings.llm_model}",
        "vector_db": settings.vector_db,
        "embedding_model": settings.embedding_model,
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/test-llm")
async def test_llm():
    """Quick test to verify Groq API is working."""
    from app.api.core.llm import get_llm_response
    reply = get_llm_response([
        {"role": "user", "content": "Reply with exactly: Groq is working."}
    ])
    return {"response": reply}

@app.get("/test-embeddings")
async def test_embeddings():
    """Quick test to verify embedding model is working."""
    from app.api.core.embeddings import embed_query
    vector = embed_query("test sentence")
    return {
        "embedding_dim": len(vector),
        "status": "embeddings working"
    }