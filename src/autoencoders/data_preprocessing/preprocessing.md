# `data_preprocessing` — how it works

This package implements Stage 1 of the data pipeline (see `plan.md`, "Data pipeline —
finalized design"): turning the raw scraped `dataset/` folder into a single, clean,
deduplicated, task-agnostic image set under `data/processed/{train,val,test}`. It's
consumed by both the denoising and colourization models — no task-specific corruption
happens here; that's applied later, on-the-fly, during training.

## Entry point

```python
from autoencoders.data_preprocessing import build_dataset

stats = build_dataset()  # defaults: DATASET_DIR -> PROCESSED_DATA_DIR
```

This is the only function the package exports. Everything else is an internal stage.

## Data flow

```
dataset/<topic>/<file>
        |
        v
has_image_extension              (extension_filter.py)
        |  bool
        v
check_duplicate_and_integrity   (dedup_and_integrity.py)
        |  image, reason
        v
validate_and_normalize          (validate_and_normalize.py)
        |  image, reason
        v
resize_and_crop                 (resize_and_crop.py)
        |  image
        v
check_blank                     (blank_check.py)
        |  image, reason
        v
save_image                      (save.py)
        |  writes PNG to data/processed/_all/
        v
   ... loop repeats for every source file ...
        v
split_dataset                   (split.py)
        |  moves files into train/val/test, deletes _all/
        v
data/processed/{train,val,test}/*.png
```

`pipeline.py`'s `build_dataset()` is the orchestrator: it just walks `dataset/`, calls
each per-image stage in order, and stops early (recording a drop reason) the moment a
stage rejects the image. No filtering logic lives in `pipeline.py` itself — it should
read as the shape of the pipeline, not its implementation.

### The stage contract

Every per-image filtering stage takes an image (or, for the first two stages, a file
path) and returns a `(result, reason)` tuple:
- `(image, None)` — passed, keep going.
- `(None, "some_reason")` — dropped; `pipeline.py` records the reason via `stats.drop(...)`
  and moves on to the next source file.

Two exceptions: `has_image_extension` returns a plain `bool` (there's no image object yet
at that point - just a path), and `resize_and_crop` is a pure transform (nothing gets
dropped by resizing/cropping), so it just returns the transformed image directly.

## Files

| File | Function | What it does |
|---|---|---|
| `extension_filter.py` | `has_image_extension(path)` | Filters by filename extension before any file I/O happens at all. Runs first since dedup/integrity both require reading the full file into memory - no point paying that cost for files that clearly aren't wanted: non-images (stray `.json`/`.DS_Store`/leftover zips, etc.) or GIFs (`.gif` deliberately excluded from `IMAGE_EXTENSIONS` - animated, not representative static photos). This subsumes what used to be a separate post-decode GIF format check. |
| `dedup_and_integrity.py` | `check_duplicate_and_integrity(path, seen_hashes)` | MD5-hashes the raw file bytes to drop exact duplicates (cheap, runs before any decode), then opens and force-decodes the image to drop unreadable/corrupt/truncated files. |
| `validate_and_normalize.py` | `validate_and_normalize(image)` | Drops images with an aspect ratio outside `MIN_ASPECT_RATIO`–`MAX_ASPECT_RATIO` (banners/panoramas) and images smaller than `MIN_SOURCE_DIM`, then converts survivors to RGB — compositing any transparency onto white rather than discarding the alpha channel naively. |
| `resize_and_crop.py` | `resize_and_crop(image)` | Resizes the shorter side to `IMG_SIZE` (preserving aspect ratio, no distortion), then center-crops to a square. Pure transform, nothing dropped. |
| `blank_check.py` | `check_blank(image)` | Drops blank/near-blank images. Computed on the *final cropped image*, not the original, since that's what training actually sees — grayscale std-dev below `BLANK_STD_THRESHOLD` is treated as flat/placeholder content. |
| `save.py` | `save_image(image, source_path, staging_dir, used_names)` | Writes the surviving image as a PNG into the flat staging folder (`data/processed/_all/`), naming it `<topic>__<original_stem>.png` with a numeric suffix on collision. |
| `split.py` | `split_dataset(staging_dir, output_dir, seed)` | Runs once, after every image has been processed (the final kept count isn't known until then). Shuffles deterministically (seeded) and moves files into `train`/`val`/`test` per `TRAIN_SPLIT`/`VAL_SPLIT` from `config.py`, then removes the now-empty staging folder. |
| `stats.py` | `CleaningStats` | Counters returned by `build_dataset()`: `scanned`, `kept`, `dropped` (a reason -> count dict, so new filters don't require editing this file), and `split_counts` (filled in after the split step). |
| `pipeline.py` | `build_dataset(source_dir, output_dir)` | The orchestrator described above. |

## Not part of this package

The colorization-only grayscale-source filter (dropping clean images with near-zero
saturation, since they have no real color to recover) is intentionally **not** here — it
only applies to colourization's training data, not denoising's, so it will live as a
separate script that runs against `data/processed/` afterward rather than being baked
into this shared cleaning pass. See `plan.md` for details.
