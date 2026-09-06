"""
Centralized configuration for the entire ML pipeline.

Every notebook and every module under ml/src/ imports paths and
hyperparameters from HERE instead of hard-coding them. This is the single
place to change a batch size, an image resolution, or a dataset location.

Path resolution strategy
-------------------------
1. If running inside a Kaggle notebook (``/kaggle/input`` exists), dataset
   directories are auto-discovered from the mounted inputs by matching
   folder-name keywords (see ``src.utils.kaggle_utils``).
2. Otherwise (local laptop), we fall back to ``ml/datasets/<name>/`` and
   expect the user to have downloaded a small subset there (see
   ``ml/README.md`` for the exact Kaggle CLI commands).
3. Nothing here silently invents a path. If a dataset cannot be located,
   the values below resolve to ``None`` and the loading functions in
   ``src/data/discovery.py`` raise a clear, actionable error the moment
   someone actually tries to read files from a missing dataset — not at
   import time (so simply importing this module never crashes).
"""
import os
from pathlib import Path

from src.utils.kaggle_utils import is_kaggle_env, find_dataset_dir

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
IS_KAGGLE = is_kaggle_env()

# Repo-relative paths (works both locally and when the ml/ folder is used
# as a Kaggle "utility script" dataset).
ML_ROOT = Path(__file__).resolve().parents[1]          # .../ml
DATASETS_ROOT = ML_ROOT / "datasets"                     # local fallback root
METADATA_DIR = DATASETS_ROOT / "metadata"                 # CSVs written by Phase 0
CHECKPOINTS_DIR = ML_ROOT / "checkpoints"
REPORTS_DIR = ML_ROOT / "reports"                           # figures, confusion matrices, etc.

METADATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset search keywords — used to auto-locate each dataset whether it's
# mounted under /kaggle/input/<slug-derived-name> or manually placed locally.
# ---------------------------------------------------------------------------
IMAGE_DATASET_KEYWORDS = ["real-and-fake-faces", "140k", "real_and_fake_faces"]
IMAGE_DATASET_SLUG = "xhlulu/140k-real-and-fake-faces"
IMAGE_LOCAL_DIR = DATASETS_ROOT / "images"

SIGNATURE_DATASET_KEYWORDS = ["signature"]
SIGNATURE_DATASET_SLUG = "emrahaydemr/realfake-signature-datasets"
SIGNATURE_LOCAL_DIR = DATASETS_ROOT / "signatures"

VIDEO_DATASET_KEYWORDS = ["deep-fake-detection", "dfd"]
VIDEO_DATASET_SLUG = "sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset"
VIDEO_LOCAL_DIR = DATASETS_ROOT / "videos"

# ---------------------------------------------------------------------------
# Resolved dataset directories (auto-discovered once, reused everywhere).
# These calls are cheap (a directory walk at most a few levels deep) and are
# safe to run at import time; discovery failures raise, they don't invent data.
# ---------------------------------------------------------------------------
IMAGE_DATASET_DIR = find_dataset_dir(
    keywords=IMAGE_DATASET_KEYWORDS, local_fallback=IMAGE_LOCAL_DIR
)
IMAGE_DATA_DIR = IMAGE_DATASET_DIR  # root containing the actual image files/folders
IMAGE_TRAIN_CSV = METADATA_DIR / "image_train_split.csv"
IMAGE_VALID_CSV = METADATA_DIR / "image_val_split.csv"
IMAGE_TEST_CSV = METADATA_DIR / "image_test_split.csv"

SIGNATURE_DATASET_DIR = find_dataset_dir(
    keywords=SIGNATURE_DATASET_KEYWORDS, local_fallback=SIGNATURE_LOCAL_DIR
)
SIGNATURE_IMAGE_DIR = SIGNATURE_DATASET_DIR
SIGNATURE_METADATA_CSV = METADATA_DIR / "signature_metadata.csv"

VIDEO_DATASET_DIR = find_dataset_dir(
    keywords=VIDEO_DATASET_KEYWORDS, local_fallback=VIDEO_LOCAL_DIR
)

# ---------------------------------------------------------------------------
# Subset sizes (Phase 0 dataset-preparation rules — see project brief)
# ---------------------------------------------------------------------------
IMAGE_SUBSET_PER_CLASS = 2000        # 2000 real + 2000 fake = 4000 total
IMAGE_SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / val / test

VIDEO_SUBSET_PER_CLASS = 50          # 50 real + 50 fake = 100 total
VIDEO_FRAMES_PER_VIDEO = 8            # sampled, not every frame

# ---------------------------------------------------------------------------
# Model / training hyperparameters — deliberately small for an 8GB-RAM laptop
# with a single consumer GPU. Override via environment variables if needed.
# ---------------------------------------------------------------------------
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 4))          # 3-5 epochs per project brief
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-5))

# Pretrained ViT backbone (small variant keeps memory/compute low)
VIT_BACKBONE = os.getenv("VIT_BACKBONE", "vit_tiny_patch16_224")

# ---------------------------------------------------------------------------
# Class taxonomies (kept centralized so label order never drifts between
# training, evaluation, and the FastAPI inference services)
# ---------------------------------------------------------------------------
IMAGE_CLASSES = ["real", "fake"]
SIGNATURE_CLASSES = ["human_genuine", "human_forged", "ai_generated"]
VIDEO_CLASSES = ["real", "fake"]


def summary() -> dict:
    """Small helper notebooks can call to print/verify the resolved config."""
    return {
        "is_kaggle": IS_KAGGLE,
        "image_dataset_dir": str(IMAGE_DATASET_DIR) if IMAGE_DATASET_DIR else None,
        "signature_dataset_dir": str(SIGNATURE_DATASET_DIR) if SIGNATURE_DATASET_DIR else None,
        "video_dataset_dir": str(VIDEO_DATASET_DIR) if VIDEO_DATASET_DIR else None,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "vit_backbone": VIT_BACKBONE,
    }
