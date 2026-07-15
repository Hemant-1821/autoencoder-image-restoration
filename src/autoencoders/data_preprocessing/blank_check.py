"""Stage 1, step 8: drop blank/near-blank images (flat color, placeholders, etc.).

Runs on the final resized+cropped image, not the original - what matters is whether the
actual training crop is informative, not the source image as a whole (see plan.md).
"""

from __future__ import annotations

from PIL import Image, ImageStat

from autoencoders.config import BLANK_STD_THRESHOLD


def check_blank(image: Image.Image) -> tuple:
    stddev = ImageStat.Stat(image.convert("L")).stddev[0]
    if stddev < BLANK_STD_THRESHOLD:
        return None, "blank"
    return image, None
