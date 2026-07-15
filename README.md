# RGA Chatbot

## Run locally

Use the Streamlit-only entry point:

```bash
python run_app.py
```

This starts the app on port 8501.

## Notes

- The app uses the local RAG service and falls back to retrieved source excerpts when the configured model endpoint is unavailable.
- For model-generated answers, configure the PrivateMode-compatible environment variables before launching.
