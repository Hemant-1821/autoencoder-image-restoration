"""Stage 1, steps 4-6: aspect ratio + minimum size filters, then normalize to RGB.

Aspect ratio and size are checked first since both are cheap, header-level properties
(no decode needed) - no point normalizing an image we're about to drop anyway.
"""

from __future__ import annotations

from PIL import Image

from autoencoders.config import MAX_ASPECT_RATIO, MIN_ASPECT_RATIO, MIN_SOURCE_DIM


def _to_rgb(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white instead of naively discarding the alpha channel."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def validate_and_normalize(image: Image.Image) -> tuple:
    width, height = image.size

    aspect_ratio = width / height
    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        return None, "aspect_ratio"

    if min(width, height) < MIN_SOURCE_DIM:
        return None, "undersized"

    return _to_rgb(image), None
