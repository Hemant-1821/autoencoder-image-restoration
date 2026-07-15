"""Orchestrates the offline cleaning stages: dataset/ -> data/processed/.

See plan.md ("Data pipeline - finalized design", Stage 1) and preprocessing.md in this
package for the full data-flow writeup. Each stage lives in its own module here and takes
the previous stage's image, either passing it on or dropping it with a reason - this
function just shows the shape of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from autoencoders.config import DATASET_DIR, PROCESSED_DATA_DIR, SEED
from autoencoders.data_preprocessing.blank_check import check_blank
from autoencoders.data_preprocessing.dedup_and_integrity import check_duplicate_and_integrity
from autoencoders.data_preprocessing.extension_filter import has_image_extension
from autoencoders.data_preprocessing.resize_and_crop import resize_and_crop
from autoencoders.data_preprocessing.save import save_image
from autoencoders.data_preprocessing.split import split_dataset
from autoencoders.data_preprocessing.stats import CleaningStats
from autoencoders.data_preprocessing.validate_and_normalize import validate_and_normalize

STAGING_DIR_NAME = "_all"


def build_dataset(
    source_dir: Path = DATASET_DIR, output_dir: Path = PROCESSED_DATA_DIR
) -> CleaningStats:
    staging_dir = output_dir / STAGING_DIR_NAME
    staging_dir.mkdir(parents=True, exist_ok=True)

    stats = CleaningStats()
    seen_hashes: set = set()
    used_names: set = set()

    for source_path in sorted(source_dir.rglob("*")):
        if not source_path.is_file():
            continue
        stats.scanned += 1

        if not has_image_extension(source_path):
            stats.drop("unsupported_extension")
            continue

        image, reason = check_duplicate_and_integrity(source_path, seen_hashes)
        if image is None:
            stats.drop(reason)
            continue

        image, reason = validate_and_normalize(image)
        if image is None:
            stats.drop(reason)
            continue

        image = resize_and_crop(image)

        image, reason = check_blank(image)
        if image is None:
            stats.drop(reason)
            continue

        save_image(image, source_path, staging_dir, used_names)
        stats.kept += 1

    stats.split_counts = split_dataset(staging_dir, output_dir, seed=SEED)
    return stats
