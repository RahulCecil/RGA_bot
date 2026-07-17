import os
import sys
from typing import List, Optional
import logging

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.database import ensure_schema
from app.services.rag_service import RAGService

app = FastAPI(title="EU AI Act RAG API", version="1.0.0")
service = RAGService()
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup_init() -> None:
    try:
        ensure_schema()
    except Exception as exc:
        # Keep API alive for health checks while surfacing init failures in logs.
        logger.warning("Database schema initialization failed on startup: %s", exc)


class ChatRequest(BaseModel):
    query: str
    limit: Optional[int] = 4


class SourceResponse(BaseModel):
    article: str
    paragraph: Optional[str] = None
    page: Optional[int] = None
    text: str


class TokenUsageResponse(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class MetricsResponse(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


class JudgeResponse(BaseModel):
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    citation_integrity_score: float = 0.0
    hallucinations_detected: List[str] = Field(default_factory=list)
    citation_errors: List[str] = Field(default_factory=list)
    explanation: str = "Judge unavailable"


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]
    token_usage: TokenUsageResponse
    metrics: MetricsResponse
    judge: JudgeResponse


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
        token_usage=TokenUsageResponse(**(result.get("token_usage") or {})),
        metrics=MetricsResponse(**{
            "prompt_tokens": getattr(result.get("metrics"), "prompt_tokens", 0),
            "completion_tokens": getattr(result.get("metrics"), "completion_tokens", 0),
            "total_tokens": getattr(result.get("metrics"), "total_tokens", 0),
            "bytes_in": getattr(result.get("metrics"), "bytes_in", 0),
            "bytes_out": getattr(result.get("metrics"), "bytes_out", 0),
        }),
        judge=JudgeResponse(**(result.get("judge") or {})),
    )
