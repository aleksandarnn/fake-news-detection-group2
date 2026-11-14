"""Guard the numbers in the README against silent drift.

Run after `python -m src.baselines --quick`:

    python -m tests.check_metrics

Fails if a metric moves outside its tolerance, if a split changes size, or if
the README points at a figure that is not on disk. Needs no test framework.
"""

from __future__ import annotations

import json
import re
import sys

from src.data import EXPERIMENTS_DIR, FIGURES_DIR, LABELS, PROJECT_ROOT, load_splits

# --------------------------------------------------------------------------
# Expected values, as reported in README.md.
#
# Tolerances are per-model on purpose. Most of these reproduce bit-for-bit;
# the SGD model does not, because hinge-loss `SGDClassifier` is sensitive to
# version-level changes in shuffling and the convergence check, so a seed does
# not pin it across scikit-learn releases. Its wider band reflects measured
# drift, not a lower standard.
# --------------------------------------------------------------------------
EXPECTED: dict[str, dict] = {
    "liar_logreg_metrics.json": {
        "name": "TF-IDF word + LogReg",
        "macro_f1": 0.2418,
        "accuracy": 0.2431,
        "tol": 0.010,
    },
    "liar_char_svm_metrics.json": {
        "name": "TF-IDF char + LinearSVC",
        "macro_f1": 0.2383,
        "accuracy": 0.2463,
        "tol": 0.010,
    },
    "liar_wordchar_sgd_metrics.json": {
        "name": "TF-IDF word+char + SGD",
        "macro_f1": 0.2435,
        "accuracy": 0.2534,
        "tol": 0.025,  # version-sensitive, see note above
    },
    "liar_metadata_only_metrics.json": {
        "name": "Metadata-only control",
        "macro_f1": 0.2541,
        "accuracy": 0.2518,
        "tol": 0.010,
    },
    "liar_ablation_drop_last_metrics.json": {
        "name": "Drop-last-sentence ablation",
        "macro_f1": 0.2379,
        "accuracy": 0.2399,
        "tol": 0.010,
    },
    # Not regenerated in CI — torch is not installed there. Checked as a
    # committed artifact so the README cannot drift away from it.
    "bert_liar_plain/metrics.json": {
        "name": "DistilBERT (fine-tuned)",
        "macro_f1": 0.2798,
        "accuracy": 0.2936,
        "tol": 0.001,
        "committed_only": True,
    },
}

EXPECTED_SPLITS = {"train": 10240, "valid": 1284, "test": 1267}

# The central claim of the README: a model with no access to the claim text
# outscores every text-only linear model. If that ever stops being true, the
# write-up needs rewriting, so fail loudly rather than quietly.
TEXT_ONLY_LINEAR = [
    "liar_logreg_metrics.json",
    "liar_char_svm_metrics.json",
    "liar_wordchar_sgd_metrics.json",
]

failures: list[str] = []
checks = 0


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg: str) -> None:
    global checks
    checks += 1
    print(f"  ok    {msg}")


def read_metrics(rel: str) -> dict | None:
    path = EXPERIMENTS_DIR / rel
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    return None


def check_splits() -> None:
    print("\nSplit sizes")
    train, val, test = load_splits()
    for name, df in (("train", train), ("valid", val), ("test", test)):
        expected = EXPECTED_SPLITS[name]
        if len(df) != expected:
            fail(f"{name}: {len(df)} rows, expected {expected}")
        else:
            ok(f"{name}: {len(df)} rows")

    missing = set(LABELS) - set(train["label"].unique())
    if missing:
        fail(f"labels missing from train: {sorted(missing)}")
    else:
        ok(f"all {len(LABELS)} labels present")


def check_metrics() -> dict[str, float]:
    print("\nMetrics")
    observed: dict[str, float] = {}
    for rel, spec in EXPECTED.items():
        d = read_metrics(rel)
        if d is None:
            if spec.get("committed_only"):
                fail(f"{spec['name']}: {rel} is missing from the repository")
            else:
                fail(f"{spec['name']}: {rel} not found — run `python -m src.baselines --quick`")
            continue

        f1 = get(d, "test_macro_f1", "macro_f1")
        acc = get(d, "test_accuracy", "accuracy")
        tol = spec["tol"]

        if f1 is None:
            fail(f"{spec['name']}: no macro-F1 in {rel}")
            continue
        observed[rel] = f1

        delta = abs(f1 - spec["macro_f1"])
        if delta > tol:
            fail(
                f"{spec['name']}: macro-F1 {f1:.4f}, "
                f"expected {spec['macro_f1']:.4f} (±{tol}) — off by {delta:.4f}"
            )
        else:
            ok(f"{spec['name']}: macro-F1 {f1:.4f} (±{delta:.4f})")

        if acc is not None:
            delta_acc = abs(acc - spec["accuracy"])
            if delta_acc > tol:
                fail(
                    f"{spec['name']}: accuracy {acc:.4f}, "
                    f"expected {spec['accuracy']:.4f} (±{tol})"
                )
    return observed


def check_headline_claim(observed: dict[str, float]) -> None:
    print("\nHeadline claim")
    meta = observed.get("liar_metadata_only_metrics.json")
    text_scores = {k: v for k, v in observed.items() if k in TEXT_ONLY_LINEAR}

    if meta is None or not text_scores:
        fail("cannot check the metadata-shortcut claim — metrics missing")
        return

    best_text = max(text_scores.values())
    if meta > best_text:
        ok(
            f"metadata-only ({meta:.4f}) still beats the best text-only "
            f"linear model ({best_text:.4f})"
        )
    else:
        fail(
            f"metadata-only ({meta:.4f}) no longer beats the best text-only "
            f"linear model ({best_text:.4f}) — the README's main finding has changed"
        )


def check_readme_figures() -> None:
    print("\nFigures referenced by the README")
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        fail("README.md is missing")
        return

    text = readme.read_text(encoding="utf-8")
    refs = set(re.findall(r'(?:src=")(figures/[^"]+)"', text))
    refs |= set(re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", text))

    if not refs:
        fail("README references no figures — did the paths change?")
        return

    for ref in sorted(refs):
        if (PROJECT_ROOT / ref).exists():
            ok(ref)
        else:
            fail(f"{ref} is referenced by the README but not on disk")


def check_no_large_files() -> None:
    print("\nRepository hygiene")
    limit_mb = 10
    offenders = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        size_mb = path.stat().st_size / 1_048_576
        if size_mb > limit_mb:
            offenders.append((path.relative_to(PROJECT_ROOT), size_mb))

    # Only flag files that git is actually tracking; .gitignore'd artifacts
    # on a developer's disk are fine.
    import subprocess

    try:
        tracked = set(
            subprocess.run(
                ["git", "ls-files"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split("\n")
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  skip  git not available")
        return

    tracked_offenders = [
        (p, s) for p, s in offenders if p.as_posix() in tracked
    ]
    if tracked_offenders:
        for p, s in tracked_offenders:
            fail(f"{p} is tracked and {s:.0f} MB — should it be gitignored?")
    else:
        ok(f"no tracked file over {limit_mb} MB")


def main() -> int:
    print("=" * 64)
    print("Checking committed results against README.md")
    print("=" * 64)

    check_splits()
    observed = check_metrics()
    check_headline_claim(observed)
    check_readme_figures()
    check_no_large_files()

    print("\n" + "=" * 64)
    if failures:
        print(f"FAILED — {len(failures)} problem(s), {checks} check(s) passed\n")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASSED — all {checks} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
