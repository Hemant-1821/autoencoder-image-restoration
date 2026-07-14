# Autoencoder Image Restoration — Plan

Two convolutional autoencoders trained on a shared pool of scraped photos:
- **Denoising**: clean image -> synthetically noised image -> reconstruct clean.
- **Colourization**: clean image -> synthetically desaturated (grayscale) -> reconstruct colour.

A Streamlit app will let a user upload any image and run it through either model.

## Data strategy (confirmed)
Both tasks draw from the same `dataset/` pool of clean photos. No separate noisy/gray
source data is needed — noise and desaturation are generated synthetically at data-prep
or training time. `data/raw/denoising` and `data/raw/colorization` can be built from the
same underlying images (e.g. same split, task-specific corruption applied on load, or
pre-generated pairs — TBD when we design the pipeline).

## Data pipeline — finalized design

Two stages: an **offline cleaning pass** (run once, output cached to disk) and an
**on-the-fly `tf.data` pipeline** (runs every training/eval step).

### Stage 1 — Offline cleaning (`dataset/` -> `data/processed/`)
Runs once over the 6,486 scraped images, produces a single canonical clean RGB dataset
shared by both tasks (no separate copies for denoising vs colorization).

1. **Integrity check**: open every file with PIL; drop unreadable/corrupt/truncated images.
2. **Format handling**:
   - Convert everything to RGB (handles RGBA/CMYK/palette/grayscale source images,
     strips alpha).
   - GIFs (71 files): drop entirely (animated, not representative static photos) —
     default; revisit if we want the extra ~1% of data.
3. **Dedup**: MD5 hash of raw file bytes to drop exact duplicates (some overlap is likely
   given overlapping topic keywords, e.g. "Cat"/"Cats", "Car"/"Cars", "Motorcycle"/
   "Motorbike"). Perceptual hashing (catches re-compressed/resized re-uploads of the same
   photo) is a possible follow-up if exact-hash dedup leaves visible near-duplicates.
4. **Resize + crop to 128x128** (confirmed, data-driven): sample of 300 images showed
   median resolution ~1024x760 and aspect ratios ranging 0.41–3.88 (only ~14% roughly
   square), so a direct crop-only approach would discard nearly all image content and
   risk cutting off the subject. Instead: **resize shorter side to 128px (preserving
   aspect ratio, no distortion) then center-crop to 128x128.**
5. **Drop images still undersized after resize** — negligible (~1/300 in the sample was
   below 128px in any dimension pre-resize).
6. **Save** as clean RGB (PNG, lossless — avoids re-introducing JPEG artifacts into what's
   supposed to be the "clean" ground truth) into `data/processed/`.
7. **Train/val/test split**: default **80/10/10**, single shared split used by both tasks
   (split once on the clean image pool, since both tasks derive from the same source
   images). Split done with `SEED=42` from `config.py` for reproducibility.

### Stage 2 — On-the-fly `tf.data` pipeline (every epoch, from `data/processed/`)
No task-specific data is pre-generated or stored — corruption is applied live so the
model sees fresh variation every epoch (train split) while eval stays reproducible
(val/test split, fixed seed).

- **Denoising**: input = clean image + Gaussian noise; target = clean image.
  - Train: noise sigma re-randomized per batch/epoch (diversity/robustness).
  - Val/test: noise generated once with a fixed seed so PSNR/SSIM are comparable run to
    run and across checkpoints.
  - Noise model: **Gaussian only** (no speckle/salt-and-pepper/blur variants).
- **Colorization**: input = L channel; target = a/b channels (Lab color space, not raw
  RGB→gray→RGB). RGB→Lab implemented as pure TF tensor ops (linear-RGB conversion +
  3x3 matrix to XYZ + elementwise nonlinearity to Lab) — not `cv2`/`skimage` wrapped in
  `tf.numpy_function`, since that runs eagerly per-image under the GIL, can't be graph
  traced/optimized or placed on GPU/Metal, and needs per-image Python calls instead of one
  vectorized batch op. Lab conversion is deterministic, so no train/val split distinction
  needed here — it's identical every time either way.
- Both pipelines read from the same `data/processed/{train,val,test}` split; task-specific
  transforms are applied per-task in the `tf.data.Dataset.map()` step, not baked into
  storage.

### Defaults assumed above (flag if you want these changed)
- Dedup via exact MD5 hash only (not perceptual hash) for the first pass.
- GIFs dropped rather than sampling a frame.
- 80/10/10 train/val/test split, shared across both tasks.
- Cleaned images stored as PNG (lossless) rather than re-compressed JPEG.

## Status

### Done
- [x] Repo scaffolding: `pyproject.toml`, `requirements.txt`/`requirements.lock.txt`.
- [x] `src/autoencoders/config.py` — paths + constants (`IMG_SIZE=(128,128)`,
      `BATCH_SIZE=32`, `SEED=42`, model output paths).
- [x] Empty stub packages: `src/autoencoders/denoising/`, `src/autoencoders/colorization/`.
- [x] Directory layout: `data/raw/{denoising,colorization}`, `data/processed`,
      `models/{denoising,colorization}`, `app/`, `scripts/`, `notebooks/` (all placeholders).
- [x] `.gitignore` tuned to keep the empty data/model dirs tracked via `.gitkeep` while
      ignoring actual data/model artifacts.
- [x] Bing scraper notebook (`bingdownloadimagespython.ipynb`) + topic list
      (`data_file.json`, 74 topics).
- [x] Ran the scraper: `dataset/` now has 6,486 images across 74 topic folders (~1.9GB,
      mixed jpg/png/jpeg/webp/gif/jfif). This folder is gitignored (local raw artifact,
      not committed).

### Left to do
- [ ] **Data pipeline**: implement the design above — offline cleaning script
      (`dataset/` -> `data/processed/{train,val,test}`) plus the on-the-fly `tf.data`
      transforms (noise for denoising, RGB→Lab for colorization). Design is finalized; not
      yet written.
- [ ] **Model architecture**: design the U-Net(s) for each task — depth (number of
      down/up-sampling stages), filter counts per stage, still to be decided. Architecture
      style (U-Net) and independence (no shared backbone) are settled — see Resolved
      decisions.
- [ ] **Training scripts**: loss functions (MSE/MAE for denoising; MSE or perceptual loss
      for colourization — possibly in Lab colour space instead of RGB), metrics (PSNR/SSIM),
      checkpointing to `models/{denoising,colorization}/model.keras`.
- [ ] **Evaluation**: qualitative (before/after grids) + quantitative (PSNR/SSIM on held-out
      test set) for both models.
- [ ] **Streamlit app** (`app/`): upload an image, run through denoising and/or
      colourization model, display results.
- [ ] **`scripts/`**: CLI entry points wrapping the data pipeline / training / evaluation.
- [ ] **`notebooks/`**: currently empty — exploratory EDA/prototyping notebooks, if wanted.
- [ ] No `README.md` yet — worth adding once the pipeline/model shape is decided.

## Open decisions (not yet made)
- U-Net depth (number of down/up-sampling stages) and filter counts per stage — TBD when
  we design the networks in detail.

## Resolved decisions
- Model architecture: **U-Net** (encoder-decoder with skip connections) for both tasks,
  not a plain bottleneck autoencoder. Rationale: a plain bottleneck is the right choice
  for anomaly detection (forcing lossy reconstruction so novel patterns reconstruct
  poorly), but this project's goal is faithful, pixel-aligned restoration — skip
  connections let the decoder combine low-level detail (edges/texture) from the encoder
  with bottleneck context at every resolution, which matters for both denoising (preserve
  real detail while removing noise) and colourization (keep colour boundaries aligned to
  edges, avoid bleeding).
- Colourization target colour space: **Lab** (predict a/b channels from L), not raw RGB.
- Data pipeline shape: offline clean/resize/crop pass cached to `data/processed/`, then
  task-specific corruption (noise, Lab split) applied on-the-fly in `tf.data` — see
  "Data pipeline — finalized design" above.
- Denoising noise model: **Gaussian only** (no speckle/salt-and-pepper/blur variants).
- Denoising and colourization are **fully independent models**, no shared backbone.
  Rationale: the two objectives conflict (denoising suppresses high-frequency detail,
  colourization needs to preserve it to place colour boundaries correctly), risking
  negative transfer in a shared encoder; dataset/model scale here is small enough that
  the parameter savings of sharing aren't worth the added training/debugging coupling.
