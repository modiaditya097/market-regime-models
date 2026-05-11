import subprocess
import sys
from pathlib import Path


def run_pipeline(run_command: list[str], project_root: Path) -> subprocess.Popen:
    """Spawn the model pipeline as a subprocess.

    Returns a Popen object. Caller is responsible for reading stdout and
    checking returncode.

    If the first arg is the literal string ``python`` (or ``python.exe``), it
    is replaced with ``sys.executable`` so the subprocess inherits the same
    interpreter (and therefore the same venv / installed packages) as the
    Shiny app. This avoids ``ModuleNotFoundError`` when the system ``python``
    on PATH differs from the one running the dashboard.

    Args:
        run_command: Command list, e.g. ``["python", "main.py"]``.
        project_root: Working directory for the subprocess.
    """
    cmd = list(run_command)
    if cmd and Path(cmd[0]).name.lower() in {"python", "python.exe"}:
        cmd[0] = sys.executable
    return subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
