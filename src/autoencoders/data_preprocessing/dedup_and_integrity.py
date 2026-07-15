"""Stage 1, steps 1-2: drop exact duplicates (MD5 of raw bytes) and unreadable/corrupt images.

Dedup runs before the decode attempt since it's the cheaper check and doesn't require
opening the image at all.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image


def check_duplicate_and_integrity(
    source_path: Path, seen_hashes: set
) -> tuple:
    raw_bytes = source_path.read_bytes()

    file_hash = hashlib.md5(raw_bytes).hexdigest()
    if file_hash in seen_hashes:
        return None, "duplicate"
    seen_hashes.add(file_hash)

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except Exception:
        return None, "unreadable"

    return image, None
