from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.generation.rag_pipeline import answer_query, answer_query_streaming

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_rewrite: bool = True
    use_rerank: bool = True 

@router.post("/ask")
async def ask(payload: QueryRequest):
    """
    Ask a question and get a full answer with citations (non-streaming).
    """
    result = answer_query(
        query=payload.query,
        top_k=payload.top_k,
        use_rewrite=payload.use_rewrite,
        use_rerank=payload.use_rerank 
    )
    return result


@router.post("/ask-stream")
async def ask_stream(payload: QueryRequest):
    """
    Ask a question and stream the answer token by token.
    Citations are sent as a final marker in the stream.
    """
    def generate():
        for chunk in answer_query_streaming(
            query=payload.query,
            top_k=payload.top_k,
            use_rewrite=payload.use_rewrite,
            use_rerank=payload.use_rerank 
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")