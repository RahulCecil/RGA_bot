# RGA Chatbot

## Run locally

Use the FastAPI entry point:

```bash
python run_app.py
```

This starts the API on port 8011 by default.

Equivalent explicit command:

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8011
```

## Streamlit frontend

Start the API first, then run:

```bash
streamlit run app/streamlit_app.py
```

The frontend calls FastAPI at `http://127.0.0.1:8011` by default.
Set `API_BASE_URL` to target a different backend URL.

## API endpoints

- `GET /health`: health check
- `POST /chat`: query endpoint

`/chat` returns:

- `answer`: generated response
- `sources`: retrieved context chunks
- `token_usage`: prompt/completion/total tokens
- `metrics`: token counts plus `bytes_in` and `bytes_out`
- `judge`: faithfulness, relevance, citation integrity, and diagnostics

Example request:

```bash
curl -X POST "http://127.0.0.1:8011/chat" ^
	-H "Content-Type: application/json" ^
	-d "{\"query\":\"What does Article 5 say about prohibited AI practices?\",\"limit\":4}"
```

## Docker

Run API + Streamlit + PostgreSQL/pgvector + LLM proxy:

```bash
copy .env.example .env
docker compose up --build
```

Open:

- API: `http://127.0.0.1:8011`
- Streamlit frontend: `http://127.0.0.1:8501`
- Postgres/pgvector: `localhost:5433`
- LLM proxy: `http://127.0.0.1:8080`

By default, API and ingestion use the proxy service inside Compose:

```bash
PRIVATEMODE_PROXY_URL=http://llm_proxy:8080/v1
```

The proxy container uses:

```bash
ghcr.io/edgelesssys/privatemode/privatemode-proxy:latest --apiKey <PRIVATEMODE_API_KEY>
```

If you prefer keeping proxy outside Compose, override in `.env`:

```bash
PRIVATEMODE_PROXY_URL=http://host.docker.internal:8080/v1
```

Run one-off ingestion/vectorization job:

```bash
docker compose --profile ingest run --rm ingest
```

Environment overrides are in `docker-compose.yml` under the `api` service.

## Notes

- The app uses the local RAG service and falls back to retrieved source excerpts when the configured model endpoint is unavailable.
- For model-generated answers, configure the PrivateMode-compatible environment variables before launching.
- Token usage and judge output are returned in the `/chat` API response payload.
