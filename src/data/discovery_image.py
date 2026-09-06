"""
Discovery + metadata-building for the "140k Real and Fake Faces" dataset
(xhlulu/140k-real-and-fake-faces).

Known layout on Kaggle (as mounted under /kaggle/input/...):

    <root>/
      real_vs_fake/real_vs_fake/train/{real,fake}/*.jpg
      real_vs_fake/real_vs_fake/valid/{real,fake}/*.jpg
      real_vs_fake/real_vs_fake/test/{real,fake}/*.jpg
      train.csv, valid.csv, test.csv   (columns vary by dataset version)

We do NOT assume the exact nesting or CSV column names are stable across
dataset versions — instead we walk the directory tree looking for `real/`
and `fake/` folders (the one thing that's guaranteed by the dataset's
description) and build our own metadata from the filesystem. This is more
robust than trusting a specific CSV schema that might not match what's
actually mounted.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetNotFoundError(RuntimeError):
    """Raised when a required dataset directory could not be located."""


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def find_class_dirs(root: Path) -> dict[str, list[Path]]:
    """
    Walk `root` and collect every directory literally named 'real' or 'fake'
    (case-insensitive). The 140k dataset nests these under train/valid/test,
    but we deliberately don't care about that split — Phase 0 rebuilds our
    own stratified split from a fresh random sample, per the project brief.
    """
    found: dict[str, list[Path]] = {"real": [], "fake": []}
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() in found:
            found[path.name.lower()].append(path)
    return found


def build_image_metadata(image_dataset_dir: Optional[Path]) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [filepath, label] where label is
    'real' or 'fake', built directly from the filesystem.

    Raises DatasetNotFoundError with an actionable message if the dataset
    directory is missing or no real/fake folders could be found inside it.
    """
    if image_dataset_dir is None:
        raise DatasetNotFoundError(
            "Image dataset not found. On Kaggle, add the dataset "
            "'xhlulu/140k-real-and-fake-faces' to this notebook's inputs. "
            "Locally, download a subset with:\n"
            "  kaggle datasets download -d xhlulu/140k-real-and-fake-faces "
            "-p ml/datasets/images && unzip -q ml/datasets/images/*.zip -d ml/datasets/images"
        )

    root = Path(image_dataset_dir)
    class_dirs = find_class_dirs(root)

    if not class_dirs["real"] or not class_dirs["fake"]:
        raise DatasetNotFoundError(
            f"Could not find both 'real/' and 'fake/' folders under {root}. "
            "Found real dirs: "
            f"{[str(p) for p in class_dirs['real']]}, fake dirs: "
            f"{[str(p) for p in class_dirs['fake']]}. Inspect the mounted "
            "dataset structure manually — it may differ from the expected layout."
        )

    rows = []
    for label, dirs in class_dirs.items():
        for d in dirs:
            for f in d.iterdir():
                if f.is_file() and _is_image(f):
                    rows.append({"filepath": str(f), "label": label})

    if not rows:
        raise DatasetNotFoundError(
            f"'real'/'fake' folders were found under {root} but contained no "
            "image files. The dataset may not have downloaded/extracted correctly."
        )

    df = pd.DataFrame(rows).drop_duplicates(subset="filepath").reset_index(drop=True)
    return df
