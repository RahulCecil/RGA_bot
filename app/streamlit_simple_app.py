import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import streamlit as st

from app.services.rag_service import RAGService


@st.cache_resource
def get_service() -> RAGService:
    return RAGService()


st.set_page_config(page_title="EU AI Act RAG Chat", page_icon="⚖️")
st.title("EU AI Act RAG Chat")
st.caption("Streamlit-only chat interface using the local RAG service.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about the EU AI Act")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching the document and generating an answer..."):
        service = get_service()
        result = service.answer_query(prompt, limit=4)

    answer = result.get("answer", "")
    judge = result.get("judge") or {}
    token_usage = result.get("token_usage") or {}

    if not isinstance(judge, dict):
        judge = {}
    if not isinstance(token_usage, dict):
        token_usage = {}

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
        if judge.get("explanation"):
            st.write(f"Explanation: {judge.get('explanation')}")
        else:
            st.write("Explanation: Judge unavailable")

        st.subheader("Sources")
        for source in result.get("sources", []):
            article = source.get("article", "Unknown")
            paragraph = source.get("paragraph") or "n/a"
            page = source.get("page", "?")
            with st.expander(f"{article} • paragraph {paragraph} • page {page}"):
                st.write(source.get("text", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
