import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / "rga_bot.venv" / "Scripts" / "python.exe"


def main() -> None:
    streamlit_port = 8501
    command = [
        str(VENV_PYTHON),
        "-m",
        "streamlit",
        "run",
        "app/streamlit_simple_app.py",
        "--server.port",
        str(streamlit_port),
    ]

    print(f"Starting Streamlit app on http://127.0.0.1:{streamlit_port}...")
    subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    main()
