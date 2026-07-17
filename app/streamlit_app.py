import os

import requests
import streamlit as st


def get_api_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8011").rstrip("/")


def ask_backend(prompt: str, limit: int = 4) -> dict:
    response = requests.post(
        f"{get_api_base_url()}/chat",
        json={"query": prompt, "limit": limit},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def render_response_diagnostics(result: dict) -> None:
    token_usage = result.get("token_usage") or {}
    metrics = result.get("metrics") or {}
    judge = result.get("judge") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Prompt tokens", token_usage.get("prompt_tokens", 0))
    col2.metric("Completion tokens", token_usage.get("completion_tokens", 0))
    col3.metric("Total tokens", token_usage.get("total_tokens", 0))

    with st.expander("Metrics", expanded=False):
        st.json(metrics)

    with st.expander("Judge", expanded=False):
        st.metric("Faithfulness", judge.get("faithfulness_score", 0.0))
        st.metric("Relevance", judge.get("relevance_score", 0.0))
        st.metric("Citation integrity", judge.get("citation_integrity_score", 0.0))
        st.write(judge.get("explanation", "Judge unavailable"))

        hallucinations = judge.get("hallucinations_detected") or []
        citation_errors = judge.get("citation_errors") or []

        st.write("Hallucinations detected:")
        if hallucinations:
            for item in hallucinations:
                st.markdown(f"- {item}")
        else:
            st.write("None")

        st.write("Citation errors:")
        if citation_errors:
            for item in citation_errors:
                st.markdown(f"- {item}")
        else:
            st.write("None")


st.set_page_config(page_title="RGA Chat", page_icon="📚")
st.title("RGA Chat: EU AI Act")
st.caption("Simple Streamlit frontend connected to the FastAPI backend.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.write(f"Backend: {get_api_base_url()}")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about the EU AI Act")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_backend(prompt)
                answer = result.get("answer", "")
                st.markdown(answer)
                st.markdown("---")
                render_response_diagnostics(result)

                sources = result.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            article = source.get("article", "Unknown")
                            paragraph = source.get("paragraph") or "n/a"
                            page = source.get("page", "?")
                            text = source.get("text", "")
                            st.markdown(f"- **{article}** | paragraph {paragraph} | page {page}")
                            st.write(text)
            except Exception as exc:
                answer = f"Request failed: {exc}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
