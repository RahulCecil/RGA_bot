import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import requests
import streamlit as st

from app.services.rag_service import RAGService


def _use_fastapi_mode() -> bool:
    return os.getenv("RGA_USE_FASTAPI", "0").strip().lower() in {"1", "true", "yes"}


@st.cache_resource
def _get_service() -> RAGService:
    return RAGService()


def _resolve_backend_url() -> str:
    configured = os.getenv("API_BASE_URL")
    if configured:
        return configured.rstrip("/")

    api_port = os.getenv("API_PORT", "8011")
    return f"http://127.0.0.1:{api_port}"


def _query_via_fastapi(prompt: str, limit: int = 4) -> dict:
    base_url = _resolve_backend_url()
    response = requests.post(
        f"{base_url}/chat",
        json={"query": prompt, "limit": limit},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _query_direct(prompt: str, limit: int = 4) -> dict:
    service = _get_service()
    return service.answer_query(prompt, limit=limit)


st.set_page_config(page_title="EU AI Act RAG Chat", page_icon="⚖️")
st.title("EU AI Act RAG Chat")
st.caption("Ask questions about the EU AI Act using local RAG retrieval and citation output.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Controls")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pop("last_result", None)

    fastapi_mode = _use_fastapi_mode()
    st.caption(f"Execution mode: {'FastAPI backend' if fastapi_mode else 'Direct Streamlit service'}")
    if fastapi_mode:
        st.caption(f"Backend URL: {_resolve_backend_url()}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about the EU AI Act")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching the document and generating an answer..."):
        try:
            if _use_fastapi_mode():
                result = _query_via_fastapi(prompt, limit=4)
            else:
                result = _query_direct(prompt, limit=4)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.session_state.messages.append({"role": "assistant", "content": "The request failed. Check your service setup and try again."})
            st.stop()

    answer = result.get("answer", "")
    judge = result.get("judge") or {}
    token_usage = result.get("token_usage") or {}

    if not isinstance(judge, dict):
        judge = {}
    if not isinstance(token_usage, dict):
        token_usage = {}

    st.session_state.last_result = result

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.markdown("---")

        st.subheader("Token usage")
        st.metric("Prompt tokens", token_usage.get("prompt_tokens", 0))
        st.metric("Completion tokens", token_usage.get("completion_tokens", 0))
        st.metric("Total tokens", token_usage.get("total_tokens", 0))

        st.subheader("Judge score")
        st.metric("Faithfulness", judge.get("faithfulness_score", 0.0))
        st.metric("Relevance", judge.get("relevance_score", 0.0))
        st.write(f"Explanation: {judge.get('explanation', 'Judge unavailable')}")

        st.subheader("Sources")
        for source in result.get("sources", []):
            article = source.get("article", "Unknown")
            paragraph = source.get("paragraph") or "n/a"
            page = source.get("page", "?")
            with st.expander(f"{article} - paragraph {paragraph} - page {page}"):
                st.write(source.get("text", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
