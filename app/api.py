import os
import sys
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.rag_service import RAGService

app = FastAPI(title="EU AI Act RAG API", version="1.0.0")
service = RAGService()


class ChatRequest(BaseModel):
    query: str
    limit: Optional[int] = 4


class SourceResponse(BaseModel):
    article: str
    paragraph: Optional[str] = None
    page: Optional[int] = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]
    token_usage: dict
    judge: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        result = service.answer_query(request.query, limit=request.limit or 4)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backend error: {exc}") from exc

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=[
            SourceResponse(
                article=source.get("article", "Unknown"),
                paragraph=source.get("paragraph") or "n/a",
                page=source.get("page"),
                text=source.get("text", ""),
            )
            for source in result.get("sources", [])
        ],
        token_usage=result.get("token_usage", {}),
        judge=result.get("judge", {}),
    )
