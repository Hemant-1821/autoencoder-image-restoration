"""Stage 1, step 7: resize the shorter side to IMG_SIZE (preserving aspect ratio), then
center-crop to a square.

A direct crop from source resolution would discard nearly all image content given how
varied the scraped images' resolutions/aspect ratios are (see plan.md). Pure transform -
nothing gets dropped here, so unlike the filter stages this returns a single value.
"""

from __future__ import annotations

from PIL import Image

from autoencoders.config import IMG_SIZE


def resize_and_crop(image: Image.Image) -> Image.Image:
    size = IMG_SIZE[0]
    width, height = image.size
    scale = size / min(width, height)
    new_width, new_height = round(width * scale), round(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - size) // 2
    top = (new_height - size) // 2
    return image.crop((left, top, left + size, top + size))
