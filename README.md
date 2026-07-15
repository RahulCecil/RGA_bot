# RGA Chatbot

## Run locally

Use the Streamlit entry point (direct mode by default):

```bash
python run_app.py
```

This starts the app on port 8501.

## Execution modes

- Default: direct Streamlit mode (no FastAPI required). The UI calls `RAGService` directly.
- Optional: FastAPI mode. Set `RGA_USE_FASTAPI=1` and run the API separately.

Example FastAPI start command:

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8011
```

## One EXE per mode (Windows)

This repository now includes two dedicated launchers that install prerequisites, run setup, and launch:

- `launch_direct_mode.py`: direct Streamlit mode
- `launch_fastapi_mode.py`: FastAPI + Streamlit mode

Build both EXEs:

```bash
python build_mode_exes.py
```

Generated executables:

- `dist/RGA_DirectMode.exe`
- `dist/RGA_FastAPIMode.exe`

Runtime behavior:

- Both launchers ensure required Python packages are installed.
- Both attempt schema setup (`ensure_schema`) before launch.
- FastAPI mode starts API, waits for `/health`, then starts Streamlit.

## Notes

- The app uses the local RAG service and falls back to retrieved source excerpts when the configured model endpoint is unavailable.
- For model-generated answers, configure the PrivateMode-compatible environment variables before launching.
- Token usage and judge output are shown in the Streamlit response panel.
