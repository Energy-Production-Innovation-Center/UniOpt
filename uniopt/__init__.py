import sys
from pathlib import Path
from subprocess import run


def get_version() -> str:
    # Try to execute version script (local development)
    if sys.platform.lower().startswith("linux"):
        script_path = Path("scripts/version.sh")
        if script_path.is_file():
            try:
                process = run("scripts/version.sh", capture_output=True, check=True)
                if process.returncode == 0:
                    return process.stdout.decode().strip()
            except Exception:
                pass
    # At last, read .version file
    version_path = Path("scripts/.version")
    if version_path.is_file():
        with version_path.open("r") as f:
            return f.read().strip()
    # Fallback value
    return "0.0.1"


__version__ = get_version()
