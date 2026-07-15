"""Stage 1, step 9: save a surviving image into the flat staging folder with a unique name."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def save_image(image: Image.Image, source_path: Path, staging_dir: Path, used_names: set) -> Path:
    topic_slug = source_path.parent.name.replace(" ", "_")
    base = f"{topic_slug}__{source_path.stem}"

    name = f"{base}.png"
    suffix = 1
    while name in used_names:
        name = f"{base}_{suffix}.png"
        suffix += 1
    used_names.add(name)

    output_path = staging_dir / name
    image.save(output_path, format="PNG")
    return output_path
