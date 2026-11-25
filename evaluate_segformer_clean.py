import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "clean_dataset"
OUTPUT_DIR = ROOT / "results" / "SegFormer_Clean"
CHECKPOINT_PATH = OUTPUT_DIR / "best_segformer.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 2
THRESHOLD = 0.5
EXPECTED_CLASSES = ["damage", "no_damage"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class EfficientAttention(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.0)

        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim,
                dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.sr_norm = None

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape

        q = self.q(x).reshape(
            batch, tokens, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)

        if self.sr is not None:
            spatial = x.transpose(1, 2).reshape(
                batch, channels, height, width
            )
            spatial = self.sr(spatial)
            spatial = spatial.flatten(2).transpose(1, 2)
            spatial = self.sr_norm(spatial)
        else:
            spatial = x

        kv = self.kv(spatial).reshape(
            batch, -1, 2, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)

        output = attention @ v
        output = output.transpose(1, 2).reshape(
            batch, tokens, channels
        )

        return self.proj(output)


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, height, width):
        batch, tokens, _ = x.shape

        x = self.fc1(x)
        x = self.act(x)

        spatial = x.transpose(1, 2).reshape(
            batch, -1, height, width
        )
        spatial = self.dwconv(spatial)
        x = spatial.flatten(2).transpose(1, 2)

        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SegFormerBlock(nn.Module):
    def __init__(self, dim, heads, sr_ratio, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        # Keep "attention" exactly as named in the saved checkpoint.
        self.attention = EfficientAttention(dim, heads, sr_ratio)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, dim * mlp_ratio)

    def forward(self, x, height, width):
        x = x + self.attention(
            self.norm1(x), height, width
        )
        x = x + self.mlp(
            self.norm2(x), height, width
        )
        return x


class SegFormerClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        dims = [32, 64, 160, 256]
        heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]

        self.embeddings = nn.ModuleList([
            OverlapPatchEmbedding(3, dims[0], 7, 4),
            OverlapPatchEmbedding(dims[0], dims[1], 3, 2),
            OverlapPatchEmbedding(dims[1], dims[2], 3, 2),
            OverlapPatchEmbedding(dims[2], dims[3], 3, 2),
        ])

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                SegFormerBlock(
                    dims[i], heads[i], sr_ratios[i]
                )
                for _ in range(2)
            ])
            for i in range(4)
        ])

        self.final_norm = nn.LayerNorm(dims[-1])
        self.head = nn.Sequential(
            nn.Linear(dims[-1], 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        for stage in range(4):
            x, height, width = self.embeddings[stage](x)

            for block in self.blocks[stage]:
                x = block(x, height, width)

            if stage < 3:
                x = x.transpose(1, 2).reshape(
                    x.shape[0], -1, height, width
                )

        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def save_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(EXPECTED_CLASSES, rotation=20)
    ax.set_yticklabels(EXPECTED_CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("SegFormer-inspired Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center"
            )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_training_curves():
    history_path = OUTPUT_DIR / "training_history.json"

    if not history_path.exists():
        print("Training history not found; skipping training curves.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list) or not history:
        print("Training history is not a non-empty list.")
        return

    epochs = [
        row.get("epoch", index + 1)
        for index, row in enumerate(history)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if all("train_loss" in row for row in history):
        axes[0].plot(
            epochs,
            [row["train_loss"] for row in history],
            label="Train loss",
        )

    if all("val_loss" in row for row in history):
        axes[0].plot(
            epochs,
            [row["val_loss"] for row in history],
            label="Validation loss",
        )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    if axes[0].lines:
        axes[0].legend()

    if all("val_macro_f1" in row for row in history):
        axes[1].plot(
            epochs,
            [row["val_macro_f1"] for row in history],
            label="Validation macro-F1",
        )

    if all("val_accuracy" in row for row in history):
        axes[1].plot(
            epochs,
            [row["val_accuracy"] for row in history],
            label="Validation accuracy",
        )

    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    if axes[1].lines:
        axes[1].legend()

    fig.suptitle("SegFormer-inspired Training Curves")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "training_curves.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    test_path = DATA_DIR / "test"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(
        str(test_path),
        transform=transform,
    )

    if test_dataset.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class order: {test_dataset.classes}; "
            f"expected {EXPECTED_CLASSES}"
        )

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Device:", DEVICE)
    print("Classes:", test_dataset.classes)
    print("Test images:", len(test_dataset))
    print("Loading existing checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = SegFormerClassifier(num_classes=1).to(DEVICE)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    y_true_list = []
    p_no_damage_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)

            # ImageFolder assigns damage=0 and no_damage=1.
            p_no_damage = torch.sigmoid(
                logits.float()
            ).reshape(-1)

            y_true_list.extend(
                labels.cpu().numpy().astype(int).tolist()
            )
            p_no_damage_list.extend(
                p_no_damage.cpu().numpy().tolist()
            )

    y_true = np.asarray(y_true_list, dtype=np.int64)
    p_no_damage = np.asarray(
        p_no_damage_list, dtype=np.float64
    )
    p_damage = 1.0 - p_no_damage

    # Threshold applies to class 1: no_damage.
    y_pred = (p_no_damage >= THRESHOLD).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=EXPECTED_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": "SegFormer-inspired classifier",
        "dataset": "data/clean_dataset",
        "test_samples": int(len(y_true)),
        "classes": EXPECTED_CLASSES,
        "image_size": IMAGE_SIZE,
        "threshold_on_no_damage_probability": THRESHOLD,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(
            y_true, y_pred,
            average="macro", zero_division=0,
        )),
        "macro_recall": float(recall_score(
            y_true, y_pred,
            average="macro", zero_division=0,
        )),
        "macro_f1": float(f1_score(
            y_true, y_pred,
            average="macro", zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred,
            average="weighted", zero_division=0,
        )),
        "roc_auc": float(roc_auc_score(
            y_true, p_no_damage
        )),
        "errors": int(np.sum(y_true != y_pred)),
        "confusion_matrix": cm.tolist(),
        "best_epoch": checkpoint.get("epoch"),
        "best_validation_macro_f1": checkpoint.get(
            "best_val_macro_f1"
        ),
    }

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    with open(
        OUTPUT_DIR / "classification_report.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    with open(
        OUTPUT_DIR / "classification_report.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "class", "precision", "recall",
            "f1-score", "support",
        ])

        for name in EXPECTED_CLASSES:
            row = report[name]
            writer.writerow([
                name,
                row["precision"],
                row["recall"],
                row["f1-score"],
                row["support"],
            ])

    with open(
        OUTPUT_DIR / "confusion_matrix.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Actual/Predicted"] + EXPECTED_CLASSES
        )

        for name, row in zip(EXPECTED_CLASSES, cm):
            writer.writerow([name] + row.tolist())

    save_confusion_matrix(cm)

    fpr, tpr, _ = roc_curve(y_true, p_no_damage)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr, tpr,
        label=f"ROC-AUC = {metrics['roc_auc']:.4f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (no_damage)")
    ax.set_title("SegFormer-inspired ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    with open(
        OUTPUT_DIR / "wrongly_predicted_test_images.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "actual_class",
            "predicted_class",
            "damage_probability",
            "no_damage_probability",
            "confidence",
        ])

        for i, (actual, predicted) in enumerate(
            zip(y_true, y_pred)
        ):
            if actual != predicted:
                writer.writerow([
                    test_dataset.samples[i][0],
                    EXPECTED_CLASSES[actual],
                    EXPECTED_CLASSES[predicted],
                    round(float(p_damage[i]), 6),
                    round(float(p_no_damage[i]), 6),
                    round(
                        float(max(p_damage[i], p_no_damage[i])),
                        6,
                    ),
                ])

    save_training_curves()

    print("\nCorrected evaluation complete.")
    print("The model was evaluated, not retrained.")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=EXPECTED_CLASSES,
        zero_division=0,
    ))
    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()