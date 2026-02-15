"""
Cross-platform runner for the model_trainer container.

Mirrors run.ps1 behavior:
- Generates a docker-compose.yml with only ./datas mounted into /workspace/datas
- Builds the image and runs the container interactively (bash)
- Cleans up on exit: docker compose down, remove compose file, prune images

Usage
  python run_docker.py

Requires Docker Desktop/Engine and Docker Compose v2 ("docker compose") or v1 ("docker-compose").
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path


CONTAINER_NAME = "model_trainer"
COMPOSE_FILENAME = "docker-compose.yml"


def resolve_compose_cmd() -> list[str]:
    """Return the Docker Compose command as a list suitable for subprocess.

    Prefers Docker Compose v2 ("docker compose"), falls back to v1 ("docker-compose").
    Raises RuntimeError if neither is available.
    """
    # Prefer v2: docker compose
    docker = shutil.which("docker")
    if docker is not None:
        try:
            subprocess.run([docker, "compose", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return [docker, "compose"]
        except Exception:
            pass

    # Fallback to v1: docker-compose
    dc1 = shutil.which("docker-compose")
    if dc1 is not None:
        return [dc1]

    raise RuntimeError("Docker Compose not found. Install Docker Compose v2 or v1.")


def write_compose_file(path: Path) -> None:
    """Write the minimal compose file that mounts only ./datas."""
    content = (
        "services:\n"
        f"  {CONTAINER_NAME}:\n"
        f"    container_name: {CONTAINER_NAME}\n"
        "    build:\n"
        "      context: ./\n"
        "      dockerfile: Dockerfile\n"
        "    environment:\n"
        "      - PYTHONDONTWRITEBYTECODE=1\n"
        "      - PYTHONUNBUFFERED=1\n"
        "      - NVIDIA_VISIBLE_DEVICES=all\n"
        "    runtime: nvidia\n"
        "    volumes:\n"
        "      - \"./datas:/workspace/datas\"\n"
        "    working_dir: /workspace\n"
        "    command: bash\n"
        "    stdin_open: true\n"
        "    tty: true\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    compose_path = script_dir / COMPOSE_FILENAME

    # Ensure datas exists so the bind mount is valid
    (script_dir / "datas").mkdir(parents=True, exist_ok=True)

    # Resolve compose command
    try:
        compose_cmd = resolve_compose_cmd()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    print("Generating docker-compose.yml...")
    write_compose_file(compose_path)

    def cleanup():
        print("Stopping and cleaning up Docker container...")
        try:
            subprocess.run([*compose_cmd, "-f", str(compose_path), "down"], cwd=script_dir)
        except Exception:
            pass
        if compose_path.exists():
            try:
                compose_path.unlink()
                print("Removed generated docker-compose.yml")
            except Exception:
                pass
        print("Pruning unused Docker images...")
        docker = shutil.which("docker")
        if docker:
            try:
                subprocess.run([docker, "image", "prune", "-f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    # Ensure cleanup runs on process exit and Ctrl+C (portable signals)
    maybe_signals = (
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGTERM", None),
        getattr(signal, "SIGHUP", None),   # not on Windows
        getattr(signal, "SIGBREAK", None), # Windows Ctrl+Break
    )
    for sig in filter(None, maybe_signals):
        try:
            signal.signal(sig, lambda *_: sys.exit(130))
        except Exception:
            pass

    try:
        print("Starting container via docker compose up --build...")
        proc = subprocess.Popen([*compose_cmd, "-f", str(compose_path), "up", "--build"], cwd=script_dir)
        proc.wait()
        return proc.returncode or 0
    except KeyboardInterrupt:
        return 130
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
