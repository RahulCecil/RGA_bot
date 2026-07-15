import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / "rga_bot.venv" / "Scripts" / "python.exe"


def main() -> None:
    streamlit_port = 8501
    python_exe = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    command = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.port",
        str(streamlit_port),
    ]

    print(f"Starting Streamlit app on http://127.0.0.1:{streamlit_port}...")
    subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    main()
