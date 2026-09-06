"""
Subset-sampling and stratified train/val/test splitting utilities shared by
every dataset-preparation notebook. Pure pandas/sklearn — no image or video
decoding happens here, so this stays fast and RAM-light regardless of the
full dataset's size (we only ever operate on filepaths + labels).
"""
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def build_balanced_subset(
    df: pd.DataFrame,
    label_col: str,
    n_per_class: int,
    random_state: int = 42,
    classes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Randomly samples up to `n_per_class` rows per class (without replacement).
    If a class has fewer than `n_per_class` rows available, takes all of them
    and prints a warning rather than failing or fabricating rows.
    """
    classes = classes or sorted(df[label_col].unique())
    parts = []
    for cls in classes:
        cls_df = df[df[label_col] == cls]
        take = min(n_per_class, len(cls_df))
        if take < n_per_class:
            print(
                f"[build_balanced_subset] WARNING: class '{cls}' has only "
                f"{len(cls_df)} rows available (< requested {n_per_class}). "
                f"Using all {take}."
            )
        parts.append(cls_df.sample(n=take, random_state=random_state))
    subset = pd.concat(parts, ignore_index=True)
    return subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits `df` into (train, val, test) DataFrames, stratified by `label_col`,
    using the given (train, val, test) ratios (must sum to 1.0).
    """
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    train_df, remainder_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Split the remainder proportionally into val/test
    relative_val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        remainder_df,
        train_size=relative_val_size,
        stratify=remainder_df[label_col],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
