from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
DENOISING_MODEL_PATH = MODELS_DIR / "denoising" / "model.keras"
COLORIZATION_MODEL_PATH = MODELS_DIR / "colorization" / "model.keras"

DENOISING_DATA_DIR = RAW_DATA_DIR / "denoising"
COLORIZATION_DATA_DIR = RAW_DATA_DIR / "colorization"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
