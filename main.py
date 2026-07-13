import time
from app.api.core.logging_config import logger
from fastapi import FastAPI , Request

from fastapi.middleware.cors import CORSMiddleware
from app.api.core.config import settings
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.api.query import router as query_router
from app.api.evaluation import router as evaluation_router
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles


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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round(time.time() - start_time, 3)

    logger.info(
        f"{request.method} {request.url.path} | "
        f"status={response.status_code} | {duration}s"
    )
    return response


app.include_router(documents_router)
app.include_router(search_router)
app.include_router(query_router)
app.include_router(evaluation_router)  
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")
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

@app.on_event("startup")
async def startup_event():
    logger.info("RAG Assistant starting up...")
    logger.info(f"LLM: {settings.llm_provider} / {settings.llm_model}")
    logger.info(f"Vector DB: {settings.vector_db}")
    logger.info(f"Embedding model: {settings.embedding_model}")




@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request data", "details": exc.errors()}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again."}
    )    