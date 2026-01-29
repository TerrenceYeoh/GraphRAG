"""
Graph RAG - Knowledge Graph based Retrieval Augmented Generation

Usage:
    python main.py build      # Build the graph index
    python main.py clean      # Clean graph/vector data (preserves extraction cache)
    python main.py clean-all  # Clean everything including extraction cache
    python main.py rebuild    # Rebuild graph (preserves extraction cache for speed)
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


import shutil


def run_clean(preserve_cache: bool = True) -> int:
    """Clean generated data directories.

    Args:
        preserve_cache: If True, keeps extraction_cache/ and summary_cache/ for faster rebuilds.
                       If False, removes everything including the caches.

    Removes:
        - graph_db/*.json (graph data, communities)
        - graph_db/extraction_cache/ (only if preserve_cache=False)
        - graph_db/summary_cache/ (only if preserve_cache=False)
        - chroma_db/ (community embeddings)

    Returns:
        Exit code (0 = success)
    """
    root = Path(__file__).parent
    graph_db_dir = root / "graph_db"
    chroma_db_dir = root / "chroma_db"

    # Directories to preserve when preserve_cache=True
    cache_dirs = {"extraction_cache", "summary_cache"}

    # Clean chroma_db entirely
    if chroma_db_dir.exists():
        logger.info("Removing chroma_db/...")
        shutil.rmtree(chroma_db_dir)
        logger.info("  Removed chroma_db/")
    else:
        logger.info("  chroma_db/ does not exist, skipping")

    # Clean graph_db
    if graph_db_dir.exists():
        if preserve_cache:
            # Only remove JSON files, keep cache directories
            logger.info("Cleaning graph_db/ (preserving caches)...")
            files_removed = 0
            for item in graph_db_dir.iterdir():
                if item.is_file() and item.suffix == ".json":
                    item.unlink()
                    files_removed += 1
                elif item.is_dir() and item.name not in cache_dirs:
                    shutil.rmtree(item)
                    files_removed += 1
            logger.info(f"  Removed {files_removed} items from graph_db/")

            # Report cache status
            for cache_name in cache_dirs:
                cache_dir = graph_db_dir / cache_name
                if cache_dir.exists():
                    cache_count = len(list(cache_dir.glob("*.json")))
                    logger.info(f"  Preserved {cache_name}/ ({cache_count} cached items)")
                else:
                    logger.info(f"  No {cache_name}/ found")
        else:
            # Remove entire directory
            logger.info("Removing graph_db/ (including extraction_cache/)...")
            shutil.rmtree(graph_db_dir)
            logger.info("  Removed graph_db/")
    else:
        logger.info("  graph_db/ does not exist, skipping")

    logger.info("Clean complete!")
    return 0


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


def run_rebuild() -> int:
    """Rebuild graph while preserving extraction cache.

    This is faster than a full rebuild because LLM extraction results
    are cached. Only the graph structure, communities, and vector store
    are regenerated.

    Returns:
        Exit code (0 = success)
    """
    logger.info("Rebuilding Graph RAG (preserving extraction cache)...")
    clean_result = run_clean(preserve_cache=True)
    if clean_result != 0:
        return clean_result
    return run_build()


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
    elif command == "clean":
        return run_clean(preserve_cache=True)
    elif command == "clean-all":
        return run_clean(preserve_cache=False)
    elif command == "rebuild":
        return run_rebuild()
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
