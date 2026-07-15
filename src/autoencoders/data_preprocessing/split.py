"""Stage 1, step 10: split the staged, cleaned images into train/val/test folders.

Runs once, after every image has been processed - the final kept count (needed for
correct split proportions) isn't known until dedup/filtering has finished, so this can't
happen inside the per-image loop.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from autoencoders.config import TRAIN_SPLIT, VAL_SPLIT


def split_dataset(staging_dir: Path, output_dir: Path, seed: int) -> dict:
    files = sorted(staging_dir.glob("*.png"))
    random.Random(seed).shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    split_files = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }

    counts = {}
    for split_name, split_paths in split_files.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in split_paths:
            shutil.move(str(path), str(split_dir / path.name))
        counts[split_name] = len(split_paths)

    staging_dir.rmdir()
    return counts
