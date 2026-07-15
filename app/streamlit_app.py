import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import streamlit as st
from app.services.rag_service import RAGService


st.set_page_config(page_title="EU AI Act RAG Chat", page_icon="⚖️")
st.title("EU AI Act RAG Chat")
st.caption("Ask questions about the EU AI Act using your PostgreSQL-backed retrieval pipeline.")

with st.sidebar:
    st.header("Controls")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pop("last_result", None)
    st.markdown("""
    This interface uses the same retrieval and judge pipeline as the terminal app.
    Make sure your PostgreSQL database and Privatemode proxy are running first.
    """)

if "service" not in st.session_state:
    st.session_state.service = RAGService()

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
        result = st.session_state.service.answer_query(prompt)

    answer = result.get("answer", "")
    judge = result.get("judge", {})
    token_usage = result.get("token_usage", {})
    st.session_state.last_result = result

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.markdown("---")
        st.subheader("Token usage")
        st.metric("Prompt tokens", token_usage.get("prompt_tokens", "n/a"))
        st.metric("Completion tokens", token_usage.get("completion_tokens", "n/a"))
        st.metric("Total tokens", token_usage.get("total_tokens", "n/a"))
        st.subheader("Judge score")
        st.metric("Faithfulness", judge.get("faithfulness_score", "n/a"))
        st.metric("Relevance", judge.get("relevance_score", "n/a"))
        if judge.get("explanation"):
            st.write(f"Explanation: {judge.get('explanation')}")

        st.subheader("Sources")
        for source in result.get("sources", []):
            article = source.get("article", "Unknown")
            paragraph = source.get("paragraph") or "n/a"
            page = source.get("page", "?")
            with st.expander(f"{article} • paragraph {paragraph} • page {page}"):
                st.write(source.get("text", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
