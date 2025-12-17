"""
Graph RAG - Knowledge Graph based Retrieval Augmented Generation

Usage:
    python main.py build      # Build the graph index
    python main.py backend    # Start the backend server
    python main.py frontend   # Start the frontend UI
    python main.py serve      # Start both backend and frontend
"""

import subprocess
import sys
from pathlib import Path

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level: <8} | {message}", level="INFO")


def run_build() -> int:
    """Run the graph indexing pipeline.

    Returns:
        Exit code from the subprocess (0 = success)
    """
    logger.info("Building Graph RAG index...")
    result = subprocess.run(
        [sys.executable, "-m", "src.build_graph_index"],
        cwd=Path(__file__).parent,
    )
    if result.returncode != 0:
        logger.error(f"Build failed with exit code {result.returncode}")
    return result.returncode


def run_backend() -> int:
    """Start the backend server.

    Returns:
        Exit code from the subprocess (0 = success)
    """
    logger.info("Starting Graph RAG backend...")
    result = subprocess.run(
        [sys.executable, "-m", "src.app_backend"],
        cwd=Path(__file__).parent,
    )
    return result.returncode


def run_frontend() -> int:
    """Start the frontend UI.

    Returns:
        Exit code from the subprocess (0 = success)
    """
    logger.info("Starting Graph RAG frontend...")
    result = subprocess.run(
        [sys.executable, "-m", "src.gradio_frontend"],
        cwd=Path(__file__).parent,
    )
    return result.returncode


def run_serve() -> None:
    """Start both backend and frontend."""
    import multiprocessing

    logger.info("Starting Graph RAG (backend + frontend)...")

    backend_process = multiprocessing.Process(target=run_backend)
    frontend_process = multiprocessing.Process(target=run_frontend)

    backend_process.start()
    frontend_process.start()

    try:
        backend_process.join()
        frontend_process.join()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        backend_process.terminate()
        frontend_process.terminate()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    if len(sys.argv) < 2:
        print(__doc__)
        return 0

    command = sys.argv[1].lower()

    if command == "build":
        return run_build()
    elif command == "backend":
        return run_backend()
    elif command == "frontend":
        return run_frontend()
    elif command == "serve":
        run_serve()
        return 0
    else:
        logger.error(f"Unknown command: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
