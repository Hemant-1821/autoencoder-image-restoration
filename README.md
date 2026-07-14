# Autoencoder Image Restoration

Convolutional autoencoders for image **denoising** and **colourization**, trained on a
custom scraped photo dataset, with a Streamlit app to run either model on any uploaded
image.

## About the application

This project builds two independent deep-learning models that each learn a different
image restoration task, and wraps both behind a single interactive app. The first model
takes a noisy photo and reconstructs a clean version of it; the second takes a grayscale
photo and reconstructs a plausible full-colour version. Both are U-Net-style convolutional
autoencoders — an encoder that progressively compresses an image down to a compact
representation, and a decoder that rebuilds a full-resolution image back up from it, with
skip connections carrying fine spatial detail (edges, texture) directly across from
encoder to decoder at every resolution.

Both tasks are trained on the same underlying pool of clean photographs. Rather than
needing paired "before/after" data collected from the real world, the corrupted input for
each task is generated synthetically from a clean source image — Gaussian noise is added
for the denoising model, and the image is desaturated for the colourization model — so a
single clean dataset can serve both tasks. This also means the ground truth for training
is exact by construction: the model is always scored against the exact clean image its
corrupted input was derived from.

The end goal is a small Streamlit web app where a user uploads any image and can run it
through the denoising model, the colourization model, or both, and see the result
side-by-side with the original. The two models are trained and stored independently
(`models/denoising/model.keras`, `models/colorization/model.keras`), so either can be
used on its own without the other.

This is a learning project (NCI Sem 2 Machine Learning coursework) — the emphasis
throughout is on building the full pipeline end-to-end (scraping → cleaning → training →
evaluation → app) and understanding *why* each design choice was made, not just producing
a final model.

### Technologies used

- **Python 3.10–3.12**
- **TensorFlow 2.16 / Keras** — model definition and training
- **tensorflow-metal** — GPU acceleration on Apple Silicon (macOS)
- **NumPy** — array/tensor manipulation
- **Pillow** — image I/O and basic processing
- **OpenCV (opencv-python)** — image processing utilities
- **scikit-learn** — data splitting and evaluation utilities
- **Streamlit** — interactive web app for running the trained models
- **bing_image_downloader** — scraping the training images (see below)
- **Kaggle CLI** — optional dataset ingestion tooling
- **Jupyter / Jupytext** — the scraping notebook and exploratory work
- **git** + GitHub — version control

## Data scraping and dataset overview

The training images were collected with the `bingdownloadimagespython.ipynb` notebook,
which uses the `bing_image_downloader` library to pull images from Bing Image Search for
a fixed list of search topics defined in `data_file.json` (74 topics — everyday objects,
animals, vehicles, scenes, e.g. *Polar Bears, Zebra, Lamborghini, Beaches, Landscapes,
Chessboard, Cricket, Underwater Sealife*). Each topic was capped at 100 images with the
adult content filter on, downloaded into a local `dataset/` folder (one subfolder per
topic).

The resulting raw dataset currently sits at **6,486 images across 74 topic folders
(~1.9GB)**, in a mix of formats: 5,086 JPG, 873 PNG, 265 JPEG, 190 WEBP, 71 GIF, and 1
JFIF. Resolutions and aspect ratios vary widely (a sampled check found a median resolution
around 1024×760, with aspect ratios ranging roughly 0.4–3.9), which is why the data
cleaning stage resizes the shorter side before cropping to a square, rather than cropping
directly — see `plan.md` for the full reasoning. `dataset/` is intentionally **not**
committed to git (large, regenerable binary data); it's the local input to the data
pipeline, which will produce the actual versioned training data under `data/processed/`.

## Project phases

The project is broken into five macro phases. Full design reasoning, resolved decisions,
and open questions for each are tracked in [`plan.md`](plan.md), which is the living
source of truth — this README gives the high-level picture.

| Phase | What it covers | Status |
|---|---|---|
| **1. Data pipeline** | Clean/dedupe/resize the scraped `dataset/` into a shared `data/processed/` train/val/test split, then apply task-specific corruption (Gaussian noise, RGB→Lab conversion) on-the-fly during training via `tf.data`. | Design finalized, **not yet implemented** |
| **2. Model architecture** | Two independent U-Net convolutional autoencoders (no shared backbone) — one per task. Skip connections preserve fine detail, which a plain bottleneck autoencoder (better suited to anomaly detection) would blur out. | Style decided (U-Net); depth/filter sizing **not yet decided** |
| **3. Training** | Loss functions (MSE/MAE for denoising; MSE in Lab space for colourization), metrics (PSNR/SSIM), checkpointing to `models/{denoising,colorization}/model.keras`. | Not started |
| **4. Evaluation** | Qualitative before/after image grids plus quantitative PSNR/SSIM on the held-out test split, for both models. | Not started |
| **5. Streamlit app** | User uploads an image, runs it through either or both trained models, views the result. | Not started |

**Where things stand right now:** repo scaffolding, dependency pinning, and the scraping
notebook are done, and the dataset has been scraped (6,486 raw images). The data pipeline
design (phase 1) and model architecture style (phase 2, U-Net) are fully decided but not
yet coded. Nothing has been trained yet.

## Project structure

```
.
├── app/                      # Streamlit app (not yet implemented)
├── bingdownloadimagespython.ipynb   # Bing image scraper
├── data_file.json            # Scraper topic list (74 topics)
├── dataset/                  # Raw scraped images (gitignored, local only)
├── data/
│   ├── raw/{denoising,colorization}/   # Placeholders (currently unused by the finalized pipeline design)
│   └── processed/             # Cleaned/split dataset will live here
├── models/{denoising,colorization}/    # Trained .keras models (gitignored)
├── notebooks/                 # Exploratory notebooks (empty so far)
├── scripts/                   # CLI entry points (empty so far)
├── src/autoencoders/
│   ├── config.py              # Shared paths/constants (IMG_SIZE, BATCH_SIZE, SEED, ...)
│   ├── denoising/             # Denoising model/training code (empty so far)
│   └── colorization/          # Colourization model/training code (empty so far)
├── plan.md                    # Detailed design decisions, rationale, and status tracking
├── pyproject.toml
└── requirements.txt / requirements.lock.txt
```

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nothing is runnable end-to-end yet (data pipeline and models aren't implemented) — see
`plan.md` for current status and next steps.
