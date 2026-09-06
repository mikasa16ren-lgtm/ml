"""
Kaggle-environment detection and dataset auto-discovery.

The project must run in two places without code changes:
  1. A Kaggle Notebook, where datasets are mounted read-only under
     /kaggle/input/<dataset-folder-name>/...
  2. A local laptop, where a small subset has been manually downloaded into
     ml/datasets/<name>/ (see ml/README.md for the exact CLI commands).

Nothing here hard-codes a single "the" path — every dataset directory is
located by walking known roots and matching folder-name keywords, so a
differently-named Kaggle mount (e.g. a dataset version suffix) still resolves.
"""
import os
from pathlib import Path
from typing import Iterable, Optional

KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def is_kaggle_env() -> bool:
    """True when running inside a Kaggle Notebook/Script environment."""
    return KAGGLE_INPUT_ROOT.exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def list_kaggle_inputs() -> list[Path]:
    """Top-level directories mounted under /kaggle/input, or [] if not on Kaggle."""
    if not KAGGLE_INPUT_ROOT.exists():
        return []
    return [p for p in KAGGLE_INPUT_ROOT.iterdir() if p.is_dir()]


def _matches_keywords(name: str, keywords: Iterable[str]) -> bool:
    name_lower = name.lower().replace(" ", "-").replace("_", "-")
    return any(kw.lower().replace("_", "-") in name_lower for kw in keywords)


def find_dataset_dir(
    keywords: Iterable[str],
    local_fallback: Path,
    search_roots: Optional[Iterable[Path]] = None,
) -> Optional[Path]:
    """
    Locate a dataset directory by keyword match.

    Search order:
      1. Every folder under /kaggle/input whose name matches `keywords`.
      2. Any extra `search_roots` the caller passes in.
      3. `local_fallback` (ml/datasets/<name>/) if it exists and is non-empty.

    Returns the resolved Path, or None if nothing matched (callers must
    handle None explicitly rather than assuming data is present).
    """
    candidates: list[Path] = []
    candidates.extend(list_kaggle_inputs())
    if search_roots:
        candidates.extend(Path(r) for r in search_roots)

    for candidate in candidates:
        if candidate.is_dir() and _matches_keywords(candidate.name, keywords):
            return candidate

    if local_fallback.exists() and any(local_fallback.iterdir()):
        return local_fallback

    return None


def try_kagglehub_download(slug: str) -> Optional[Path]:
    """
    Best-effort download via `kagglehub` when a dataset isn't already mounted
    (useful outside Kaggle notebooks, e.g. Colab, if the user is authenticated
    locally). Never touches or prints credentials; if kagglehub isn't
    installed or the user isn't authenticated, this simply returns None so
    the caller can fall back to instructing the user to download manually.
    """
    try:
        import kagglehub  # type: ignore
    except ImportError:
        return None

    try:
        path = kagglehub.dataset_download(slug)
        return Path(path)
    except Exception as exc:  # noqa: BLE001 - surfacing any download failure as "not available"
        print(f"[kaggle_utils] kagglehub download for '{slug}' failed: {exc}")
        return None
