"""Stage 1, step 0: skip non-image files, and GIFs, by extension before reading them
into memory.

Runs before dedup/integrity, which both require reading the full file into memory (for
hashing and decoding respectively) - no point paying that cost for files that clearly
aren't wanted: non-images (stray .json/.DS_Store/leftover zips, etc.) or animated GIFs
(not representative static photos). This check needs only the path, not the file
contents, so it costs no I/O at all - it also replaces what used to be a separate
post-decode GIF format check further down the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from autoencoders.config import IMAGE_EXTENSIONS


def has_image_extension(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS
