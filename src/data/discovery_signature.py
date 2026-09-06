"""
Discovery + metadata-building for the Real/Fake Signature dataset
(emrahaydemr/realfake-signature-datasets).

IMPORTANT — label honesty:
This dataset's folder/file naming conventions are not guaranteed to be
stable, and fields like "Gender" or "Age" that sometimes appear in signature
datasets are demographic metadata, NOT genuine/forged/AI-generated labels.
We never treat them as such.

Strategy:
1. Walk the mounted directory and collect every image file.
2. For each file, inspect its full path (all parent folder names + filename)
   for keyword tokens that plausibly indicate the THREE target classes:
     - human_genuine : "genuine", "real", "original"
     - human_forged   : "forg" (forged/forgery), "fake" without "ai"/"gan"
     - ai_generated    : "ai", "gan", "generated", "synthetic", "gpt", "diffusion"
3. Every file gets a `label` of one of the three classes, or `"unknown"` if
   no keyword matched.
4. We compute and return a reliability report (% labeled per class, % unknown).
   The notebook decides whether the reliability is high enough to train on —
   this module does NOT invent labels to force a clean dataset.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Ordered so more-specific tokens are checked before generic ones.
LABEL_KEYWORDS = {
    "ai_generated": ["ai_generated", "ai-generated", "synthetic", "diffusion", "gan", "gpt", "generated"],
    "human_forged": ["forged", "forgery", "forge", "fake"],
    "human_genuine": ["genuine", "real", "original", "authentic"],
}


class DatasetNotFoundError(RuntimeError):
    pass


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _infer_label(path: Path, root: Path) -> tuple[str, str]:
    """
    Returns (label, matched_keyword). label is 'unknown' if nothing matched.

    IMPORTANT: searches only the path *relative to `root`* (all folder names
    + filename below the dataset root), never the full absolute path. The
    absolute path can contain misleading substrings from parent directory
    names (e.g. a project folder named "...deepfake-detection-platform..."
    contains "fake", which would otherwise false-positive-match every file).
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path  # fallback, shouldn't happen given how we call this
    haystack = str(relative).lower()
    for label, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in haystack:
                return label, kw
    return "unknown", ""


def build_signature_metadata(signature_dataset_dir: Optional[Path]) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [filepath, label, matched_keyword].
    Raises DatasetNotFoundError if the directory is missing or has no images.
    Does NOT raise if labels are unreliable — that is reported, not hidden,
    via `signature_label_reliability_report`.
    """
    if signature_dataset_dir is None:
        raise DatasetNotFoundError(
            "Signature dataset not found. On Kaggle, add "
            "'emrahaydemr/realfake-signature-datasets' to this notebook's inputs. "
            "Locally, download a subset with:\n"
            "  kaggle datasets download -d emrahaydemr/realfake-signature-datasets "
            "-p ml/datasets/signatures && unzip -q ml/datasets/signatures/*.zip -d ml/datasets/signatures"
        )

    root = Path(signature_dataset_dir)
    rows = []
    for f in root.rglob("*"):
        if f.is_file() and _is_image(f):
            label, keyword = _infer_label(f, root)
            rows.append({"filepath": str(f), "label": label, "matched_keyword": keyword})

    if not rows:
        raise DatasetNotFoundError(f"No image files found anywhere under {root}.")

    return pd.DataFrame(rows).drop_duplicates(subset="filepath").reset_index(drop=True)


def signature_label_reliability_report(df: pd.DataFrame) -> dict:
    """
    Quantifies how much of the dataset got a confident label vs 'unknown'.
    The notebook prints this and only proceeds to train a 3-class model if
    the 'unknown' fraction is small; otherwise it must clearly report the
    issue instead of training on invented labels.
    """
    total = len(df)
    counts = df["label"].value_counts().to_dict()
    unknown = counts.get("unknown", 0)
    return {
        "total_files": total,
        "counts_per_label": counts,
        "unknown_count": unknown,
        "unknown_fraction": round(unknown / total, 4) if total else None,
        "reliable_enough_for_training": (unknown / total) < 0.15 if total else False,
    }
