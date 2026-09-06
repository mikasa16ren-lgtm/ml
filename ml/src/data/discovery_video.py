"""
Discovery + metadata-building for the "Deep Fake Detection (DFD) Entire
Original Dataset" (sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset).

Known layout: two top-level folders, one of "original"/"real" videos and one
of "manipulated"/"fake" videos (naming varies by mirror). We detect the label
from folder-name keywords rather than assuming an exact folder name, and we
never open/decode a video file during discovery — only `os.walk` over paths —
so this step is essentially free in RAM and time even over a large dataset.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

REAL_KEYWORDS = ["original", "real"]
FAKE_KEYWORDS = ["manipulated", "fake", "deepfake", "synthesis"]


class DatasetNotFoundError(RuntimeError):
    pass


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _infer_label(path: Path, root: Path) -> str:
    """
    Infers 'real'/'fake' from the path *relative to `root`* only — never the
    full absolute path, which can contain misleading substrings from parent
    directory names (see discovery_signature.py for a concrete example of
    this exact bug).
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    haystack = str(relative).lower()
    if any(kw in haystack for kw in FAKE_KEYWORDS):
        return "fake"
    if any(kw in haystack for kw in REAL_KEYWORDS):
        return "real"
    return "unknown"


def build_video_metadata(video_dataset_dir: Optional[Path]) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [filepath, label] for every video file
    found under `video_dataset_dir`. Cheap: only walks paths, never opens a
    video file. Raises DatasetNotFoundError if nothing is found.
    """
    if video_dataset_dir is None:
        raise DatasetNotFoundError(
            "Video dataset not found. On Kaggle, add "
            "'sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset' "
            "to this notebook's inputs. Locally, download a subset with:\n"
            "  kaggle datasets download -d sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset "
            "-p ml/datasets/videos && unzip -q ml/datasets/videos/*.zip -d ml/datasets/videos\n"
            "NOTE: this dataset is large — only keep a handful of files locally for development."
        )

    root = Path(video_dataset_dir)
    rows = [
        {"filepath": str(f), "label": _infer_label(f, root)}
        for f in root.rglob("*")
        if f.is_file() and _is_video(f)
    ]

    if not rows:
        raise DatasetNotFoundError(f"No video files found anywhere under {root}.")

    return pd.DataFrame(rows).drop_duplicates(subset="filepath").reset_index(drop=True)
