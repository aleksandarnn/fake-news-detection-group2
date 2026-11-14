"""Regenerate every figure in `figures/` from tracked artifacts.

Reads the LIAR splits and the metrics JSON files that live in `experiments/`,
so this runs in seconds and needs no trained model on disk.

    python -m src.figures
    python -m src.figures --only comparison validity

Figures that need per-example predictions (the DistilBERT confusion matrix and
per-class F1 chart) are produced by `src/distilbert.py` during evaluation and
are skipped here, since the predictions are not version-controlled.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
import pandas as pd

from .data import (
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    LABELS,
    PROJECT_ROOT,
    SHORT_LABELS,
    load_splits,
)

# One consistent look across every figure.
INK = "#1f2933"
ACCENT = "#2f6f9f"
MUTED = "#9aa5b1"
HIGHLIGHT = "#b4654a"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


def _read_metric(rel: str) -> tuple[float | None, float | None]:
    """Return (macro_f1, accuracy) from a metrics file, or (None, None)."""
    path = EXPERIMENTS_DIR / rel
    if not path.exists():
        return None, None
    d = json.loads(path.read_text(encoding="utf-8"))
    f1 = d.get("test_macro_f1", d.get("macro_f1"))
    acc = d.get("test_accuracy", d.get("accuracy"))
    return (float(f1) if f1 is not None else None,
            float(acc) if acc is not None else None)


def label_distribution(train, val, test) -> None:
    """Per-split label counts — shows the imbalance that motivates macro-F1."""
    counts = pd.DataFrame(
        {
            "Train": train["label"].value_counts(),
            "Valid": val["label"].value_counts(),
            "Test": test["label"].value_counts(),
        }
    ).reindex(LABELS)

    fig, (ax_all, ax_test) = plt.subplots(1, 2, figsize=(9.5, 3.4))

    counts.plot(
        kind="bar",
        ax=ax_all,
        color=[ACCENT, MUTED, HIGHLIGHT],
        width=0.78,
        edgecolor="none",
    )
    ax_all.set_title("Label distribution across splits")
    ax_all.set_ylabel("# claims")
    ax_all.set_xlabel("")
    ax_all.set_xticklabels(SHORT_LABELS, rotation=0)
    ax_all.legend(frameon=False, fontsize=8)

    test_counts = counts["Test"]
    bars = ax_test.bar(SHORT_LABELS, test_counts.values, color=ACCENT, edgecolor="none")
    ax_test.set_title("Test split only")
    ax_test.set_ylabel("# claims")
    for bar, value in zip(bars, test_counts.values):
        ax_test.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_test.set_ylim(0, test_counts.max() * 1.15)

    fig.suptitle("")
    fig.tight_layout()
    _save(fig, "liar_label_dist_test.png")


def statement_lengths(train) -> None:
    """Claims are short — this is the single most important fact about LIAR."""
    lengths = train["statement"].str.len()
    median = lengths.median()

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.hist(
        lengths.clip(upper=400),
        bins=45,
        color=ACCENT,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(median, color=HIGHLIGHT, linewidth=1.6, linestyle="--")
    ax.text(
        median + 8,
        ax.get_ylim()[1] * 0.88,
        f"median {median:.0f} chars",
        color=HIGHLIGHT,
        fontsize=8.5,
    )
    ax.set_title("Statement length, training split")
    ax.set_xlabel("characters (clipped at 400)")
    ax.set_ylabel("# claims")
    fig.tight_layout()
    _save(fig, "liar_length_hist_train.png")


def model_comparison() -> None:
    """All models side by side, with the published SOTA drawn in as context."""
    sources = [
        ("TF-IDF word + LogReg", "liar_logreg_metrics.json"),
        ("TF-IDF char + LinearSVC", "liar_char_svm_metrics.json"),
        ("TF-IDF word+char + SGD", "liar_wordchar_sgd_metrics.json"),
        ("Metadata-only (control)", "liar_metadata_only_metrics.json"),
        ("DistilBERT (fine-tuned)", "bert_liar_plain/metrics.json"),
    ]
    rows = []
    for name, rel in sources:
        f1, acc = _read_metric(rel)
        if f1 is not None:
            rows.append((name, f1, acc))

    if not rows:
        print("  skipped model_comparison — no metrics on disk")
        return

    df = pd.DataFrame(rows, columns=["model", "macro_f1", "accuracy"])
    df = df.sort_values("macro_f1")

    colors = [
        HIGHLIGHT if "Metadata" in m else ACCENT if "DistilBERT" in m else MUTED
        for m in df["model"]
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.barh(df["model"], df["macro_f1"], color=colors, edgecolor="none")
    for bar, value in zip(bars, df["macro_f1"]):
        ax.text(
            value + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8.5,
        )

    # Random-guess floor for a 6-class problem.
    ax.axvline(1 / 6, color=INK, linewidth=1.0, linestyle=":", alpha=0.6)
    ax.annotate(
        "chance (1/6)",
        xy=(1 / 6, len(df) - 0.4),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7.5,
        alpha=0.75,
        va="top",
    )

    ax.set_xlim(0, max(df["macro_f1"]) * 1.18)
    ax.set_xlabel("macro-F1 on the LIAR test set")
    ax.set_title("Model comparison")
    fig.tight_layout()
    _save(fig, "model_comparison_macroF1.png")


def validity_checks() -> None:
    """The diagnostics: does the model lean on a final clause, or on metadata?"""
    baseline_f1, _ = _read_metric("liar_logreg_metrics.json")
    drop_f1, _ = _read_metric("liar_ablation_drop_last_metrics.json")
    meta_f1, _ = _read_metric("liar_metadata_only_metrics.json")

    if baseline_f1 is None or meta_f1 is None:
        print("  skipped validity_checks — run `python -m src.baselines --quick` first")
        return
    if drop_f1 is None:
        print("  note: ablation metrics missing, plotting without the drop-last bar")

    names, values, colors = ["Baseline\n(full text)"], [baseline_f1], [MUTED]
    if drop_f1 is not None:
        names.append("Drop last\nsentence")
        values.append(drop_f1)
        colors.append(ACCENT)
    names.append("Metadata only\n(no text)")
    values.append(meta_f1)
    colors.append(HIGHLIGHT)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.bar(names, values, color=colors, width=0.6, edgecolor="none")

    for bar, value in zip(bars, values):
        delta = value - baseline_f1
        label = f"{value:.4f}"
        if abs(delta) > 1e-9:
            label += f"\nΔ {delta:+.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(baseline_f1, color=INK, linewidth=0.9, linestyle=":", alpha=0.5)
    ax.set_ylim(0, max(values) * 1.3)
    ax.set_ylabel("macro-F1 (test)")
    ax.set_title("Validity checks — ablation and shortcut signal")
    fig.tight_layout()
    _save(fig, "liar_validity_checks.png")


FIGURE_KEYS = ["labels", "lengths", "comparison", "validity"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--only",
        nargs="+",
        choices=FIGURE_KEYS,
        default=FIGURE_KEYS,
        help="regenerate only these figures (default: all)",
    )
    args = parser.parse_args()

    needs_data = {"labels", "lengths"} & set(args.only)
    train = val = test = None
    if needs_data:
        train, val, test = load_splits()

    print("Regenerating figures...")
    if "labels" in args.only:
        label_distribution(train, val, test)
    if "lengths" in args.only:
        statement_lengths(train)
    if "comparison" in args.only:
        model_comparison()
    if "validity" in args.only:
        validity_checks()
    print("Done.")


if __name__ == "__main__":
    main()
