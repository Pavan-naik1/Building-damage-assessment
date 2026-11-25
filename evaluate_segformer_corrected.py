
from pathlib import Path
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from tqdm import tqdm


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "clean_dataset"
CHECKPOINT_PATH = (
    ROOT / "results" / "SegFormer_Clean" / "best_segformer.pt"
)
OUTPUT_DIR = ROOT / "results" / "SegFormer_CorrectedEval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class EfficientAttention(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()

        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.0)

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
            feature_map = x.transpose(1, 2).reshape(
                batch, channels, height, width
            )
            reduced = self.sr(feature_map)
            reduced = reduced.flatten(2).transpose(1, 2)
            reduced = self.sr_norm(reduced)
        else:
            reduced = x

        kv = self.kv(reduced).reshape(
            batch,
            reduced.shape[1],
            2,
            self.heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)

        output = attention @ v
        output = output.permute(
            0, 2, 1, 3
        ).contiguous().reshape(batch, tokens, channels)

        return self.dropout(self.proj(output))


class MixFFN(nn.Module):
    """MixFFN matching the operation order in the training script."""

    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()

        hidden_dim = dim * expansion
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape

        x = self.fc1(x)
        hidden_channels = x.shape[-1]

        # Training order: depthwise convolution first.
        x = x.transpose(1, 2).reshape(
            batch, hidden_channels, height, width
        )
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        # Then activation, dropout, projection, and dropout.
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class SegFormerBlock(nn.Module):
    def __init__(self, dim, heads, sr_ratio, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attention = EfficientAttention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, expansion=4, dropout=dropout)

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
        for stage_index, (embedding, blocks) in enumerate(
            zip(self.embeddings, self.blocks)
        ):
            if stage_index > 0:
                batch, tokens, channels = x.shape
                side = int(tokens ** 0.5)

                if side * side != tokens:
                    raise RuntimeError(
                        "Token count cannot be reshaped into a square"
                    )

                x = x.transpose(1, 2).contiguous().reshape(
                    batch, channels, side, side
                )

            x, height, width = embedding(x)

            for block in blocks:
                x = block(x, height, width)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


# ============================================================
# EVALUATION
# ============================================================

def main():
    print("=" * 65)
    print("SEGFORMER CORRECTED EVALUATION")
    print("=" * 65)
    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Test data:", DATA_DIR / "test")
    print("Output directory:", OUTPUT_DIR)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    expected_classes = checkpoint.get(
        "classes", ["damage", "no_damage"]
    )
    image_size = int(checkpoint.get("image_size", 224))
    threshold = float(checkpoint.get("threshold", 0.5))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(
        str(test_dir),
        transform=transform,
    )

    print("Dataset class mapping:", test_dataset.class_to_idx)
    print("Checkpoint classes:", expected_classes)

    if test_dataset.classes != expected_classes:
        raise ValueError(
            "Class order mismatch!\n"
            f"Checkpoint classes: {expected_classes}\n"
            f"Dataset classes: {test_dataset.classes}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = SegFormerClassifier(num_classes=1).to(DEVICE)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    all_labels = []
    all_probabilities = []
    all_paths = []

    with torch.no_grad():
        for images, labels in tqdm(
            test_loader,
            desc="Evaluating test images",
        ):
            images = images.to(DEVICE)
            logits = model(images)

            # Training labels: damage=0, no_damage=1.
            no_damage_probabilities = torch.sigmoid(logits)

            all_labels.extend(labels.numpy().astype(int).tolist())
            all_probabilities.extend(
                no_damage_probabilities.cpu().numpy()
                .astype(float).tolist()
            )

    labels = np.asarray(all_labels, dtype=np.int64)
    probabilities = np.asarray(
        all_probabilities, dtype=np.float64
    )
    predictions = (probabilities >= threshold).astype(np.int64)

    metrics = {
        "model": "SegFormer-inspired classifier",
        "dataset": "data/clean_dataset",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_macro_f1": checkpoint.get(
            "best_val_macro_f1"
        ),
        "classes": expected_classes,
        "image_size": image_size,
        "threshold_on_no_damage_probability": threshold,
        "test_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(
                labels, predictions,
                average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                labels, predictions,
                average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                labels, predictions,
                average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels, predictions,
                average="weighted", zero_division=0
            )
        ),
        "errors": int(np.sum(labels != predictions)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=expected_classes,
            output_dict=True,
            zero_division=0,
        ),
    }

    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(labels, probabilities)
        )
    else:
        metrics["roc_auc"] = None

    results_path = OUTPUT_DIR / "test_metrics.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    wrong_path = OUTPUT_DIR / "wrong_predictions.csv"
    with wrong_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_path",
            "actual_class",
            "predicted_class",
            "no_damage_probability",
            "damage_probability",
            "confidence",
        ])

        for index, (actual, predicted) in enumerate(
            zip(labels, predictions)
        ):
            if actual != predicted:
                no_damage_probability = float(probabilities[index])
                damage_probability = 1.0 - no_damage_probability

                writer.writerow([
                    test_dataset.samples[index][0],
                    expected_classes[actual],
                    expected_classes[predicted],
                    no_damage_probability,
                    damage_probability,
                    max(
                        no_damage_probability,
                        damage_probability,
                    ),
                ])

    print("\nRESULTS")
    print("-" * 65)
    print("Test images:", len(labels))
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"ROC-AUC:         {metrics['roc_auc']}")
    print("Errors:", metrics["errors"])
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(np.asarray(metrics["confusion_matrix"]))
    print("\nClassification report:")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=expected_classes,
        zero_division=0,
    ))
    print("Metrics saved to:", results_path)
    print("Wrong predictions saved to:", wrong_path)


if __name__ == "__main__":
    main()
