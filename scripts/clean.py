"""
Clean up temporary files, cache, and intermediate outputs.

Usage:
    python scripts/clean.py           # Dry run (show what would be removed)
    python scripts/clean.py --execute # Actually remove files
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories to clean (relative to project root)
CLEAN_DIRS = [
    "benchmark/results",
    ".pytest_cache",
    "__pycache__",
    "*.egg-info",
]

# Patterns to clean recursively
CLEAN_PATTERNS = [
    "**/__pycache__/",
    "**/*.pyc",
    "**/*.pyo",
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/.ipynb_checkpoints/",
]


def find_targets() -> list[Path]:
    """Find all directories/files to clean."""
    targets: list[Path] = []

    # Explicit directories
    for dir_name in CLEAN_DIRS:
        target = PROJECT_ROOT / dir_name
        if target.exists():
            targets.append(target)

    # Pattern-based
    for pattern in CLEAN_PATTERNS:
        for match in PROJECT_ROOT.glob(pattern):
            targets.append(match)

    # Also clean .git-hook temporary files
    return sorted(set(targets))


def dry_run(targets: list[Path]) -> None:
    """Show what would be removed."""
    total_size = 0

    print("Would remove:")
    for target in targets:
        if target.is_file():
            size = target.stat().st_size
            total_size += size
            print(f"  {target} ({size:,} bytes)")
        elif target.is_dir():
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            total_size += size
            file_count = sum(1 for _ in target.rglob("*") if _.is_file())
            print(f"  {target}/ ({file_count} files, {size:,} bytes)")

    print(f"\nTotal: {len(targets)} items, {total_size:,} bytes")


def execute(targets: list[Path]) -> None:
    """Remove targets."""
    total_size = 0

    for target in targets:
        try:
            if target.is_file():
                total_size += target.stat().st_size
                target.unlink()
                print(f"  Removed: {target}")
            elif target.is_dir():
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                total_size += size
                shutil.rmtree(target)
                print(f"  Removed: {target}/")
        except Exception as e:
            print(f"  FAILED: {target} — {e}")

    print(f"\nCleaned: {len(targets)} items, {total_size:,} bytes freed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean project temp files")
    parser.add_argument("--execute", action="store_true", help="Actually remove files")
    args = parser.parse_args()

    targets = find_targets()

    if not targets:
        print("Nothing to clean.")
        return

    if args.execute:
        execute(targets)
    else:
        dry_run(targets)
        print("\nRun with --execute to actually remove files.")


if __name__ == "__main__":
    main()
