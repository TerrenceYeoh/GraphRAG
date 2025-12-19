#!/usr/bin/env python3
"""
Utility script to identify and retry failed chunk extractions.

Usage:
    python scripts/retry_failed_chunks.py --dry-run     # Preview which chunks would be retried
    python scripts/retry_failed_chunks.py               # Delete failed cache files (then run build)
    python scripts/retry_failed_chunks.py --list-empty  # List chunks with empty extraction results
"""

import argparse
import json
import re
from pathlib import Path


def parse_failed_chunks_from_log(log_path: Path) -> set[str]:
    """Extract chunk IDs that had extraction failures from the log file."""
    failed_chunks = set()

    # Patterns for different failure types
    patterns = [
        r"LLM extraction failed for chunk ([^:]+):",
        r"Failed to parse JSON.*chunk[_\s]+([a-f0-9]+_chunk_\d+)",
    ]

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return failed_chunks

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    failed_chunks.add(match.group(1))

    return failed_chunks


def find_empty_cache_files(cache_dir: Path) -> set[str]:
    """Find cache files that have empty entities and relationships."""
    empty_chunks = set()

    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return empty_chunks

    for cache_file in cache_dir.glob("*.json"):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            if not entities and not relationships:
                chunk_id = cache_file.stem
                empty_chunks.add(chunk_id)
        except (json.JSONDecodeError, KeyError):
            # Corrupted cache file - add to retry list
            empty_chunks.add(cache_file.stem)

    return empty_chunks


def delete_cache_files(
    cache_dir: Path, chunk_ids: set[str], dry_run: bool = False
) -> int:
    """Delete cache files for the specified chunk IDs."""
    deleted = 0

    for chunk_id in sorted(chunk_ids):
        cache_file = cache_dir / f"{chunk_id}.json"
        if cache_file.exists():
            if dry_run:
                print(f"  Would delete: {cache_file.name}")
            else:
                cache_file.unlink()
                print(f"  Deleted: {cache_file.name}")
            deleted += 1

    return deleted


def main():
    parser = argparse.ArgumentParser(
        description="Retry failed chunk extractions by clearing their cache files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--list-empty",
        action="store_true",
        help="Also include chunks with empty extraction results (0 entities)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/build_graph_index.log"),
        help="Path to the build log file",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("graph_db/extraction_cache"),
        help="Path to the extraction cache directory",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Retry Failed Chunks Utility")
    print("=" * 60)

    # Find failed chunks from log
    print(f"\nScanning log file: {args.log_file}")
    failed_from_log = parse_failed_chunks_from_log(args.log_file)
    print(f"  Found {len(failed_from_log)} failed chunks in log")

    # Optionally find empty cache files
    empty_chunks = set()
    if args.list_empty:
        print(f"\nScanning cache directory: {args.cache_dir}")
        empty_chunks = find_empty_cache_files(args.cache_dir)
        print(f"  Found {len(empty_chunks)} chunks with empty results")

    # Combine both sets
    all_failed = failed_from_log | empty_chunks

    if not all_failed:
        print("\nNo failed chunks found. Nothing to retry.")
        return

    print(f"\nTotal chunks to retry: {len(all_failed)}")

    if args.dry_run:
        print("\n[DRY RUN] Would delete these cache files:")
    else:
        print("\nDeleting cache files:")

    deleted = delete_cache_files(args.cache_dir, all_failed, args.dry_run)

    print(f"\n{'Would delete' if args.dry_run else 'Deleted'}: {deleted} cache files")

    if not args.dry_run and deleted > 0:
        print("\nNext step: Run 'python main.py build' to retry the failed extractions")
    elif args.dry_run and deleted > 0:
        print("\nTo proceed, run this script without --dry-run")


if __name__ == "__main__":
    main()
