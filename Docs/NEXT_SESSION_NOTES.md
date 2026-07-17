# Next Session Notes

## Current Working Setup
- FastAPI launcher: `python run_app.py`
- API server entry point: `app/api.py`
- Default API URL: `http://127.0.0.1:8011`

## API Endpoints
- Health: `GET /health`
- Chat: `POST /chat`

## RAG Pipeline Snapshot
- Retriever: `app/repository/pgvector_retriever.py`
- Generation with citations: `app/services/rag_service.py`
- Judge: `app/services/judge_service.py`
- Token metrics: `app/services/token_tracker.py`

## Known Gaps / Next Work
- Expose `bytes_in` and `bytes_out` in API response.
- Add configurable chunking strategy selection and evaluation output.
- Add explicit Trustworthy AI checklist output (limitations, confidence, source coverage).
- Add stronger API diagnostics and structured error envelope for failed backend calls.

## Data/Infra Notes
- Expected source PDF path: `docs/EU_AI_Act_EN_TXT.pdf`
- Embedding dimension controlled by `PRIVATEMODE_EMBEDDING_DIM` (default `2560`)
- PostgreSQL/PGVector schema is ensured by `app/core/database.py`
