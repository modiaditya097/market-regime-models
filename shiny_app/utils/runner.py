import subprocess
from pathlib import Path


def run_pipeline(run_command: list[str], project_root: Path) -> subprocess.Popen:
    """Spawn the model pipeline as a subprocess.

    Returns a Popen object. Caller is responsible for reading stdout and
    checking returncode.

    Args:
        run_command: Command list, e.g. ["python", "main.py"].
        project_root: Working directory for the subprocess.
    """
    return subprocess.Popen(
        run_command,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
