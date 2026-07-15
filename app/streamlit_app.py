import os
import subprocess
import sys
import time


def _backend_is_healthy(base_url: str) -> bool:
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"query": "health check", "limit": 1},
            timeout=10,
        )
        return response.ok
    except requests.RequestException:
        return False

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import requests
import streamlit as st


def start_backend_if_needed() -> str:
    configured = os.getenv("API_BASE_URL")
    if configured:
        return configured.rstrip("/")

    api_port = os.getenv("API_PORT", "8011")
    base_url = f"http://127.0.0.1:{api_port}"

    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.ok and _backend_is_healthy(base_url):
            return base_url
    except requests.RequestException:
        pass

    repo_root = REPO_ROOT
    python_exe = os.path.join(repo_root, "rga_bot.venv", "Scripts", "python.exe")
    if os.path.exists(python_exe):
        subprocess.Popen(
            [python_exe, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", api_port],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        time.sleep(3)

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.ok:
            return base_url
    except requests.RequestException:
        pass

    return base_url


def get_backend_url() -> str:
    configured = os.getenv("API_BASE_URL")
    if configured:
        return configured.rstrip("/")

    api_port = os.getenv("API_PORT")
    candidates = []
    if api_port:
        candidates.append(f"http://127.0.0.1:{api_port}")
    candidates.extend([
        "http://127.0.0.1:8011",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8002",
    ])

    for base_url in candidates:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.ok and _backend_is_healthy(base_url):
                return base_url
        except requests.RequestException:
            continue

    return candidates[0] if candidates else "http://127.0.0.1:8001"


st.set_page_config(page_title="EU AI Act RAG Chat", page_icon="⚖️")
st.title("EU AI Act RAG Chat")
st.caption("Ask questions about the EU AI Act using your PostgreSQL-backed retrieval pipeline.")

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = start_backend_if_needed()

with st.sidebar:
    st.header("Controls")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pop("last_result", None)

    st.caption(f"Resolved backend URL: {st.session_state.api_base_url}")
    st.caption("The UI will use this address for /health and /chat requests.")
    try:
        health_response = requests.get(
            f"{st.session_state.api_base_url}/health",
            timeout=5,
        )
        health_response.raise_for_status()
        st.success("FastAPI backend: online")
    except Exception:
        st.error("FastAPI backend: offline")

    # st.markdown("""
    #This interface uses the same retrieval and judge pipeline as the terminal app.
    # Make sure your PostgreSQL database and Privatemode proxy are running first.
    # """)

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
        try:
            response = requests.post(
                f"{st.session_state.api_base_url}/chat",
                json={"query": prompt, "limit": 4},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            st.error(f"Could not reach the API backend at {st.session_state.api_base_url}: {exc}")
            st.session_state.messages.append({"role": "assistant", "content": "The backend is unavailable or returned an error."})
            st.stop()

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
