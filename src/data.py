"""Loading and label handling for the LIAR dataset.

LIAR ships as three tab-separated files with no header and 14 columns. We keep
the official splits untouched so results stay comparable to published numbers.

Run directly to print split sizes and label counts:

    python -m src.data
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths — resolved relative to the repo root, so scripts work from anywhere.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIAR_DIR = PROJECT_ROOT / "data" / "LIAR_dataset"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGURES_DIR = PROJECT_ROOT / "figures"

SEED = 42

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
# Ordered from least to most truthful. The integer id IS the position here,
# which is what makes the confusion matrices readable: adjacent ids are
# adjacent veracity levels, so off-by-one errors cluster near the diagonal.
LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# Compact tick labels for plots.
SHORT_LABELS = ["PF", "F", "BT", "HT", "MT", "T"]

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

# Non-textual fields used only by the metadata-only control.
META_COLS = [
    "speaker",
    "party_affiliation",
    "state_info",
    "subject",
    "job_title",
    "context",
]


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy. Call before anything stochastic."""
    random.seed(seed)
    np.random.seed(seed)


def load_liar(path: str | Path) -> pd.DataFrame:
    """Load one LIAR split.

    Everything is read as `str` with `keep_default_na=False` on purpose: the
    metadata columns contain literal strings like "none" and "NA" that pandas
    would otherwise silently turn into NaN, which would change what the
    metadata-only baseline sees.

    Adds a `label_id` column. Rows whose label is not one of the six known
    values are dropped — a handful of malformed rows exist in the widely
    circulated Kaggle copy of LIAR, which is why our split sizes differ
    slightly from the official counts (see README).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing LIAR split: {path}\n"
            "Expected the three .tsv files under data/LIAR_dataset/."
        )

    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(COLUMNS)} columns, got {df.shape[1]}. "
            "This file does not look like the standard LIAR layout."
        )
    df.columns = COLUMNS

    df["label_id"] = df["label"].map(LABEL_TO_ID)
    unknown = int(df["label_id"].isna().sum())
    if unknown:
        print(f"  {path.name}: dropped {unknown} row(s) with an unrecognised label")
        df = df[df["label_id"].notna()].copy()
    df["label_id"] = df["label_id"].astype(int)

    return df.reset_index(drop=True)


def load_splits(
    liar_dir: str | Path = LIAR_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) as DataFrames."""
    liar_dir = Path(liar_dir)
    return (
        load_liar(liar_dir / "train.tsv"),
        load_liar(liar_dir / "valid.tsv"),
        load_liar(liar_dir / "test.tsv"),
    )


def metadata_text(df: pd.DataFrame) -> list[str]:
    """Flatten the metadata columns into one synthetic sentence per row.

    Used by the metadata-only control: it gives a classifier everything
    *except* what the speaker actually said.
    """
    return df[META_COLS].fillna("").astype(str).agg(" ".join, axis=1).tolist()


def drop_last_sentence(text: str) -> str:
    """Remove the final sentence of a claim.

    Used by the last-sentence ablation. Splits on sentence-ending punctuation
    and discards the trailing segment. If the claim is a single sentence it is
    returned unchanged, so the ablation measures the effect of removing a
    *final clause*, never of blanking the input entirely.
    """
    import re

    parts = re.split(r"([.!?])", text)
    if len(parts) >= 2:
        parts = parts[:-2]
    return "".join(parts).strip() if parts else text


def dataset_stats(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Split sizes and per-label counts, in the shape written to experiments/."""
    return {
        "train_shape": list(train.shape),
        "val_shape": list(val.shape),
        "test_shape": list(test.shape),
        "train_label_counts": train["label"].value_counts().to_dict(),
        "val_label_counts": val["label"].value_counts().to_dict(),
        "test_label_counts": test["label"].value_counts().to_dict(),
    }


def main() -> None:
    set_seed()
    train, val, test = load_splits()

    print(f"Train {train.shape}  Val {val.shape}  Test {test.shape}\n")

    counts = pd.DataFrame(
        {
            "train": train["label"].value_counts(),
            "valid": val["label"].value_counts(),
            "test": test["label"].value_counts(),
        }
    ).reindex(LABELS)
    print(counts.to_string())

    lengths = train["statement"].str.len()
    print(
        f"\nStatement length (train, characters): "
        f"median {lengths.median():.0f} | mean {lengths.mean():.0f} | "
        f"IQR {lengths.quantile(0.25):.0f}-{lengths.quantile(0.75):.0f} | "
        f"max {lengths.max()}"
    )

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    stats_path = EXPERIMENTS_DIR / "liar_dataset_stats.json"
    stats_path.write_text(
        json.dumps(dataset_stats(train, val, test), indent=2), encoding="utf-8"
    )

    label_map_path = EXPERIMENTS_DIR / "label_map.json"
    label_map_path.write_text(json.dumps(LABEL_TO_ID, indent=2), encoding="utf-8")

    print(f"\nWrote {stats_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {label_map_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
