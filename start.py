"""Cross-platform first-run installer and launcher for KneeAI.

Run with a supported Python interpreter:

    python start.py

The script creates an isolated virtual environment, installs pinned packages on
the first run (or when requirements change), and starts the local web app.
"""

from __future__ import print_function

import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
MARKER = VENV / ".knee-ai-requirements"


def venv_python():
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def requirement_hash():
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def run(command):
    print("+ " + " ".join(str(part) for part in command))
    subprocess.check_call([str(part) for part in command], cwd=str(ROOT))


def main():
    if sys.version_info < (3, 10):
        print("KneeAI requires Python 3.10 or newer.", file=sys.stderr)
        print("Install a current Python from https://www.python.org/downloads/ and retry.", file=sys.stderr)
        return 2

    if not venv_python().exists():
        print("Creating an isolated KneeAI environment...")
        run([sys.executable, "-m", "venv", str(VENV)])

    current_hash = requirement_hash()
    installed_hash = MARKER.read_text().strip() if MARKER.exists() else ""
    if installed_hash != current_hash:
        print("Installing pinned application packages...")
        run([venv_python(), "-m", "pip", "install", "--upgrade", "pip"])
        run([venv_python(), "-m", "pip", "install", "-r", REQUIREMENTS])
        MARKER.write_text(current_hash + "\n")

    print("Starting KneeAI. Press Ctrl+C to stop.")
    command = [
        venv_python(),
        "-m",
        "streamlit",
        "run",
        ROOT / "app.py",
        "--server.address=localhost",
        "--server.port=8501",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    try:
        return subprocess.call([str(part) for part in command], cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\nKneeAI stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
