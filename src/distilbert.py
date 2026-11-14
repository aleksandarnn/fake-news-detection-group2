"""Fine-tune DistilBERT on LIAR, text only.

A plain PyTorch loop rather than `Trainer`: it keeps the dependency list to
`torch` + `transformers`, and makes the training step readable end to end.
Tokenisation happens in the DataLoader's collate function, so each batch is
padded to its own longest sequence instead of a global maximum.

Checkpoint selection is by validation macro-F1, matching how the linear
baselines were tuned.

Usage
-----
    python -m src.distilbert                      # 3 epochs, auto CPU/GPU
    python -m src.distilbert --epochs 4 --lr 3e-5
    python -m src.distilbert --save-model         # also write the weights

Runtime: a few minutes on a GPU, roughly an hour on CPU.
Writes `experiments/bert_liar_plain/metrics.json`, `test_predictions.csv`,
and two figures.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .data import (
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    ID_TO_LABEL,
    LABELS,
    PROJECT_ROOT,
    SEED,
    SHORT_LABELS,
    load_splits,
    set_seed,
)

OUT_DIR = EXPERIMENTS_DIR / "bert_liar_plain"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument(
        "--batch-train",
        type=int,
        default=None,
        help="default: 16 on GPU, 8 on CPU",
    )
    p.add_argument(
        "--batch-eval",
        type=int,
        default=None,
        help="default: 32 on GPU, 16 on CPU",
    )
    p.add_argument(
        "--save-model",
        action="store_true",
        help="write weights + tokenizer to experiments/bert_liar_plain/model "
        "(~250 MB, gitignored)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Imported here so that `--help` and the other src modules do not pay the
    # cost of loading torch.
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_train = args.batch_train or (16 if device.type == "cuda" else 8)
    batch_eval = args.batch_eval or (32 if device.type == "cuda" else 16)
    print(f"Device: {device} | train batch {batch_train} | eval batch {batch_eval}")

    train_df, val_df, test_df = load_splits()
    print(f"Train {train_df.shape} | Val {val_df.shape} | Test {test_df.shape}")

    class TextLabelDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = list(texts)
            self.labels = np.asarray(labels, dtype=int)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.texts[idx], int(self.labels[idx])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def collate_fn(batch):
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    def loader(df, batch_size, shuffle):
        ds = TextLabelDataset(df["statement"], df["label_id"])
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )

    train_loader = loader(train_df, batch_train, True)
    val_loader = loader(val_df, batch_eval, False)
    test_loader = loader(test_df, batch_eval, False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id={v: k for k, v in ID_TO_LABEL.items()},
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    @torch.no_grad()
    def evaluate(loader_):
        model.eval()
        preds, trues, total_loss = [], [], 0.0
        for batch in loader_:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += float(out.loss.item()) * batch["labels"].size(0)
            preds.append(out.logits.argmax(dim=-1).cpu().numpy())
            trues.append(batch["labels"].cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        return (
            f1_score(trues, preds, average="macro"),
            accuracy_score(trues, preds),
            total_loss / len(loader_.dataset),
            trues,
            preds,
        )

    best_val_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += float(loss.item())
            if step % 100 == 0 or step == len(train_loader):
                print(
                    f"  epoch {epoch} | step {step}/{len(train_loader)} "
                    f"| loss {running / step:.4f}"
                )

        val_f1, val_acc, val_loss, _, _ = evaluate(val_loader)
        print(
            f"epoch {epoch}: val macro-F1 {val_f1:.4f} | "
            f"val acc {val_acc:.4f} | val loss {val_loss:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  new best (val macro-F1 {val_f1:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint: val macro-F1 {best_val_f1:.4f}")

    test_f1, test_acc, test_loss, y_true, y_pred = evaluate(test_loader)
    print(
        f"\nTEST  macro-F1 {test_f1:.4f} | accuracy {test_acc:.4f} "
        f"| loss {test_loss:.4f}\n"
    )
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4,
                                zero_division=0))

    # ---------------- artifacts ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    pd.DataFrame(
        {
            "id": test_df["id"],
            "statement": test_df["statement"],
            "true_label_id": y_true,
            "true_label": [ID_TO_LABEL[i] for i in y_true],
            "pred_label_id": y_pred,
            "pred_label": [ID_TO_LABEL[i] for i in y_pred],
        }
    ).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    metrics = {
        "macro_f1": float(test_f1),
        "accuracy": float(test_acc),
        "val_best_macro_f1": float(best_val_f1),
        "model_name": args.model_name,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_train": batch_train,
        "batch_eval": batch_eval,
        "seed": SEED,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix — DistilBERT (test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(LABELS)), SHORT_LABELS)
    ax.set_yticks(range(len(LABELS)), SHORT_LABELS)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > threshold else "#1f2933",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "liar_confusion_matrix_distilbert_plain.png", dpi=200)
    plt.close(fig)

    # Per-class F1.
    report = classification_report(
        y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0
    )
    f1s = [report[label]["f1-score"] for label in LABELS]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.bar(SHORT_LABELS, f1s, color="#2f6f9f", width=0.62, edgecolor="none")
    for bar, value in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
            f"{value:.3f}", ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(0, max(f1s) * 1.25 if max(f1s) > 0 else 1)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 — DistilBERT (test)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "liar_per_class_f1_distilbert.png", dpi=200)
    plt.close(fig)

    if args.save_model:
        model.save_pretrained(OUT_DIR / "model")
        tokenizer.save_pretrained(OUT_DIR / "model")
        print(f"Saved weights to {(OUT_DIR / 'model').relative_to(PROJECT_ROOT)}")

    print(f"Wrote metrics, predictions and figures under "
          f"{OUT_DIR.relative_to(PROJECT_ROOT)} and figures/")


if __name__ == "__main__":
    main()
