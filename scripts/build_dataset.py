#!/usr/bin/env python3
"""CLI entry point: dataset/ -> data/processed/{train,val,test}.

See plan.md ("Data pipeline - finalized design") and
src/autoencoders/data_preprocessing/preprocessing.md for how the pipeline works.
"""

from __future__ import annotations

import argparse
import shutil

from autoencoders.config import PROCESSED_DATA_DIR
from autoencoders.data_preprocessing import build_dataset

SPLIT_NAMES = ("train", "val", "test")


def _existing_split_dirs() -> list:
    return [
        name
        for name in SPLIT_NAMES
        if (PROCESSED_DATA_DIR / name).exists() and any((PROCESSED_DATA_DIR / name).iterdir())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing data/processed/{train,val,test} contents before running.",
    )
    args = parser.parse_args()

    existing = _existing_split_dirs()
    if existing and not args.force:
        print(f"data/processed already has content in: {existing}. Re-run with --force to overwrite.")
        raise SystemExit(1)
    if args.force:
        for name in SPLIT_NAMES:
            shutil.rmtree(PROCESSED_DATA_DIR / name, ignore_errors=True)

    print("Cleaning dataset...")
    stats = build_dataset()

    print(f"Scanned: {stats.scanned}")
    print(f"Kept: {stats.kept}")
    print("Dropped:")
    for reason, count in sorted(stats.dropped.items()):
        print(f"  {reason}: {count}")
    print(f"Split sizes: {stats.split_counts}")


if __name__ == "__main__":
    main()
