# Next Session Notes

## Current Working Setup
- Streamlit default launcher: `python run_app.py`
- Unified Streamlit entry point: `app/streamlit_app.py`
- Direct mode is default (`RGA_USE_FASTAPI=0`)
- FastAPI mode can be enabled (`RGA_USE_FASTAPI=1`)

## EXE Launchers (Windows)
- Direct mode launcher: `launch_direct_mode.py`
- FastAPI mode launcher: `launch_fastapi_mode.py`
- Build both EXEs: `python build_mode_exes.py`

## RAG Pipeline Snapshot
- Retriever: `app/repository/pgvector_retriever.py`
- Generation with citations: `app/services/rag_service.py`
- Judge: `app/services/judge_service.py`
- Token metrics: `app/services/token_tracker.py`

## Known Gaps / Next Work
- Expose `bytes_in` and `bytes_out` in API and Streamlit UI.
- Add configurable chunking strategy selection and evaluation output.
- Add explicit Trustworthy AI checklist output (limitations, confidence, source coverage).
- Add stronger FastAPI-mode diagnostics in UI for failed backend calls.

## Data/Infra Notes
- Expected source PDF path: `docs/EU_AI_Act_EN_TXT.pdf`
- Embedding dimension controlled by `PRIVATEMODE_EMBEDDING_DIM` (default `2560`)
- PostgreSQL/PGVector schema is ensured by `app/core/database.py`
