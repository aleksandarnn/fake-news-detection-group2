"""TF-IDF + linear baselines, plus the two diagnostic controls.

Every model is trained on `train.tsv` and scored on the untouched `test.tsv`.
Macro-F1 is the headline metric because the six classes are imbalanced
(pants-fire has ~1/3 the support of half-true), so plain accuracy would reward
a model for ignoring the rare classes.

Usage
-----
    python -m src.baselines                  # full grid search (slow: ~20-40 min)
    python -m src.baselines --quick          # reuse recorded best params (~1 min)
    python -m src.baselines --only logreg metadata
    python -m src.baselines --quick --save-models

Results are written to `experiments/<slug>_metrics.json` and collected into
`experiments/liar_models_summary.csv`.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .data import (
    EXPERIMENTS_DIR,
    LABELS,
    PROJECT_ROOT,
    SEED,
    drop_last_sentence,
    load_splits,
    metadata_text,
    set_seed,
)

# --------------------------------------------------------------------------
# Best hyper-parameters found by the full grid searches, recorded so that
# `--quick` reproduces the published numbers without re-running the search.
# --------------------------------------------------------------------------
RECORDED_BEST: dict[str, dict[str, Any]] = {
    "logreg": {"clf__C": 0.5},
    "char_svm": {
        "tfidf__ngram_range": (3, 6),
        "clf__C": 0.25,
        "clf__class_weight": "balanced",
        "clf__loss": "squared_hinge",
    },
    "wordchar_sgd": {
        "vec__word__ngram_range": (1, 2),
        "vec__char__ngram_range": (3, 5),
        "clf__alpha": 1e-3,
        "clf__penalty": "l2",
    },
}


def logreg_pipeline() -> Pipeline:
    """Word unigrams + bigrams into balanced logistic regression.

    The transparent baseline: every coefficient maps back to a readable n-gram.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000, random_state=SEED, class_weight="balanced"
                ),
            ),
        ]
    )


def char_svm_pipeline() -> Pipeline:
    """Character 3-6 grams into a linear SVM.

    Character n-grams survive typos and odd tokenisation, which matters on
    short, quote-heavy political claims.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 6),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(random_state=SEED)),
        ]
    )


def wordchar_sgd_pipeline() -> Pipeline:
    """Word and character n-grams concatenated, trained with hinge-loss SGD."""
    return Pipeline(
        [
            (
                "vec",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=1,
                                max_df=0.95,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char",
                                ngram_range=(3, 6),
                                min_df=1,
                                max_df=0.95,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", SGDClassifier(loss="hinge", random_state=SEED)),
        ]
    )


def metadata_pipeline() -> Pipeline:
    """Same classifier as the text baseline, fed only non-textual metadata."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    random_state=SEED,
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )


PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logreg": {"clf__C": [0.5, 1.0, 2.0]},
    "char_svm": {
        "tfidf__ngram_range": [(3, 5), (3, 6), (4, 6)],
        "clf__C": [0.25, 0.5, 1, 2, 4],
        "clf__class_weight": [None, "balanced"],
        "clf__loss": ["squared_hinge", "hinge"],
    },
    "wordchar_sgd": {
        "vec__word__ngram_range": [(1, 1), (1, 2)],
        "vec__char__ngram_range": [(3, 5), (3, 6)],
        "clf__alpha": [1e-5, 1e-4, 1e-3],
        "clf__penalty": ["l2", "elasticnet"],
    },
}

CV_FOLDS = {"logreg": 3, "char_svm": 5, "wordchar_sgd": 5}

DISPLAY_NAMES = {
    "logreg": "TFIDF(word 1-2) + LogReg",
    "char_svm": "TFIDF(char 3-6) + LinearSVC",
    "wordchar_sgd": "TFIDF(word 1-2 + char 3-6) + SGD(hinge)",
    "metadata": "Metadata-only",
    "ablation": "Drop-last-sentence (text)",
}

SLUGS = {
    "logreg": "liar_logreg",
    "char_svm": "liar_char_svm",
    "wordchar_sgd": "liar_wordchar_sgd",
    "metadata": "liar_metadata_only",
    "ablation": "liar_ablation_drop_last",
}

ALL_EXPERIMENTS = ["logreg", "char_svm", "wordchar_sgd", "metadata", "ablation"]


def fit_and_score(
    key: str,
    pipeline: Pipeline,
    X_train: list[str],
    y_train,
    X_test: list[str],
    y_test,
    *,
    quick: bool,
    save_model: bool,
    report: bool = False,
) -> dict:
    """Fit one pipeline (optionally via grid search) and score it on test."""
    name = DISPLAY_NAMES[key]
    grid = PARAM_GRIDS.get(key)
    started = time.perf_counter()

    if grid and not quick:
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="f1_macro",
            cv=CV_FOLDS[key],
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        best_params = RECORDED_BEST.get(key, {})
        if best_params:
            pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        model = pipeline

    y_pred = model.predict(X_test)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    accuracy = float(accuracy_score(y_test, y_pred))
    elapsed = time.perf_counter() - started

    print(f"\n{name}")
    print(f"  macro-F1 {macro_f1:.4f} | accuracy {accuracy:.4f} | {elapsed:.1f}s")
    if best_params:
        print(f"  params   {best_params}")
    if report:
        print(
            classification_report(
                y_test, y_pred, target_names=LABELS, digits=4, zero_division=0
            )
        )

    metrics = {
        "model": name,
        "test_macro_f1": macro_f1,
        "test_accuracy": accuracy,
        # Tuples are not JSON-serialisable; stringify for the record.
        "best_params": {k: str(v) for k, v in best_params.items()},
        "grid_searched": bool(grid) and not quick,
        "seed": SEED,
    }

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out = EXPERIMENTS_DIR / f"{SLUGS[key]}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  wrote    {out.relative_to(PROJECT_ROOT)}")

    if save_model:
        from joblib import dump

        model_path = EXPERIMENTS_DIR / f"{SLUGS[key]}_model.joblib"
        dump(model, model_path)
        print(f"  wrote    {model_path.relative_to(PROJECT_ROOT)} (gitignored)")

    return metrics


def write_summary() -> pd.DataFrame | None:
    """Collect every metrics file on disk into one sorted comparison table."""
    rows = []
    sources = [
        (DISPLAY_NAMES["logreg"], "liar_logreg_metrics.json"),
        (DISPLAY_NAMES["char_svm"], "liar_char_svm_metrics.json"),
        (DISPLAY_NAMES["wordchar_sgd"], "liar_wordchar_sgd_metrics.json"),
        (DISPLAY_NAMES["metadata"], "liar_metadata_only_metrics.json"),
        ("DistilBERT (fine-tuned)", "bert_liar_plain/metrics.json"),
    ]
    for name, rel in sources:
        path = EXPERIMENTS_DIR / rel
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        f1 = d.get("test_macro_f1", d.get("macro_f1"))
        acc = d.get("test_accuracy", d.get("accuracy"))
        if f1 is None:
            continue
        rows.append((name, float(f1), float(acc) if acc is not None else None))

    if not rows:
        print("\nNo metrics found — nothing to summarise.")
        return None

    summary = pd.DataFrame(rows, columns=["model", "macro_f1", "accuracy"])
    summary = summary.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    out = EXPERIMENTS_DIR / "liar_models_summary.csv"
    summary.to_csv(out, index=False)

    print("\n" + "=" * 62)
    print("Model comparison on the LIAR test set (sorted by macro-F1)")
    print("=" * 62)
    print(summary.to_string(index=False))
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip grid search and use the recorded best hyper-parameters",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_EXPERIMENTS,
        default=ALL_EXPERIMENTS,
        help="run only these experiments (default: all)",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="also dump fitted pipelines as .joblib (gitignored, ~45 MB total)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print a full per-class precision/recall/F1 report for each model",
    )
    args = parser.parse_args()

    set_seed()
    print("Loading LIAR splits...")
    train, val, test = load_splits()
    print(f"Train {train.shape} | Val {val.shape} | Test {test.shape}")
    if args.quick:
        print("\nRunning in --quick mode: recorded best params, no grid search.")

    y_train = train["label_id"].to_numpy()
    y_test = test["label_id"].to_numpy()
    text_train = train["statement"].tolist()
    text_test = test["statement"].tolist()

    builders = {
        "logreg": (logreg_pipeline, lambda: (text_train, text_test)),
        "char_svm": (char_svm_pipeline, lambda: (text_train, text_test)),
        "wordchar_sgd": (wordchar_sgd_pipeline, lambda: (text_train, text_test)),
        "metadata": (
            metadata_pipeline,
            lambda: (metadata_text(train), metadata_text(test)),
        ),
        # The ablation reuses the logistic-regression setup so the only thing
        # that changes between it and the baseline is the input text.
        "ablation": (
            logreg_pipeline,
            lambda: (
                [drop_last_sentence(t) for t in text_train],
                [drop_last_sentence(t) for t in text_test],
            ),
        ),
    }

    for key in ALL_EXPERIMENTS:
        if key not in args.only:
            continue
        build_pipeline, build_data = builders[key]
        Xtr, Xte = build_data()
        # The ablation is a control, not a tuned model: always reuse the
        # baseline's hyper-parameters so the comparison stays like-for-like.
        quick = args.quick or key == "ablation"
        pipeline = build_pipeline()
        if key == "ablation":
            pipeline.set_params(**RECORDED_BEST["logreg"])
        fit_and_score(
            key,
            pipeline,
            Xtr,
            y_train,
            Xte,
            y_test,
            quick=quick,
            save_model=args.save_models,
            report=args.report,
        )

    write_summary()


if __name__ == "__main__":
    main()
