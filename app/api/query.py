from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.generation.rag_pipeline import answer_query, answer_query_streaming
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi import Depends
from app.generation.auth import verify_api_key


router = APIRouter(prefix="/query", tags=["Query"])

MAX_QUERY_LENGTH = 500

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=5, ge=1, le=20)
    use_rewrite: bool = True
    use_rerank: bool = True

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
@router.post("/ask")
async def ask(payload: QueryRequest):
    try:
        result = answer_query(
            query=payload.query,
            top_k=payload.top_k,
            use_rewrite=payload.use_rewrite,
            use_rerank=payload.use_rerank
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/ask-stream")

@router.post("/ask", dependencies=[Depends(verify_api_key)])
async def ask(payload: QueryRequest):
    ...
async def ask_stream(payload: QueryRequest):
    def generate():
        try:
            for chunk in answer_query_streaming(
                query=payload.query,
                top_k=payload.top_k,
                use_rewrite=payload.use_rewrite,
                use_rerank=payload.use_rerank
            ):
                yield chunk
        except Exception as e:
            yield f"\n\nError: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")