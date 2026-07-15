from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"  # raw scraped images (bing_image_downloader output)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
DENOISING_MODEL_PATH = MODELS_DIR / "denoising" / "model.keras"
COLORIZATION_MODEL_PATH = MODELS_DIR / "colorization" / "model.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# Offline cleaning pipeline (see plan.md, "Data pipeline - finalized design")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}  # static image extensions to keep; .gif excluded (animated)
MIN_SOURCE_DIM = 64  # drop images whose shorter side is below this rather than upscale heavily
MIN_ASPECT_RATIO = 0.5  # drop images (width/height) outside [MIN_ASPECT_RATIO, MAX_ASPECT_RATIO]
MAX_ASPECT_RATIO = 2.0  # banners/panoramas would otherwise be destructively cropped
BLANK_STD_THRESHOLD = 10  # grayscale pixel std-dev (0-255) below this on the final crop = blank/near-blank
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Colorization-only filter (separate script, not part of the shared cleaning pipeline)
COLORIZATION_MIN_SATURATION = 5  # mean HSV saturation (0-255) below this = treat source as already-grayscale
