import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / "rga_bot.venv" / "Scripts" / "python.exe"


def main() -> None:
    api_host = os.getenv("API_HOST", "127.0.0.1")
    api_port = os.getenv("API_PORT", "8011")
    python_exe = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    command = [
        python_exe,
        "-m",
        "uvicorn",
        "app.api:app",
        "--host",
        api_host,
        "--port",
        str(api_port),
        "--reload",
    ]

    print(f"Starting FastAPI app on http://{api_host}:{api_port}...")
    subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    main()
