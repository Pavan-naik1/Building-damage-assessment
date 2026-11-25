from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_DIR / "data" / "clean_dataset"
OUTPUT_DIR = PROJECT_DIR / "results" / "SegFormer_Clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
SEED = 42
THRESHOLD = 0.5

EXPECTED_CLASSES = ["damage", "no_damage"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print("=" * 60)
print("CLEAN-DATASET SEGFORMER-INSPIRED CLASSIFIER")
print("=" * 60)
print("Dataset:", DATA_ROOT)
print("Output:", OUTPUT_DIR.resolve())
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM GB:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    )

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Clean dataset not found: {DATA_ROOT}\n"
        "Expected data/clean_dataset/train, validation, and test."
    )


# ============================================================
# DATA
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_dataset(folder_name, transform):
    folder = DATA_ROOT / folder_name

    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    dataset = datasets.ImageFolder(str(folder), transform=transform)

    if dataset.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected classes in {folder_name}: {dataset.classes}. "
            f"Expected: {EXPECTED_CLASSES}"
        )

    return dataset


train_dataset = load_dataset("train", train_transform)
val_dataset = load_dataset("validation", eval_transform)
test_dataset = load_dataset("test", eval_transform)

print("\nClasses:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
)


# ============================================================
# SEGFORMER-INSPIRED ARCHITECTURE
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
        batch, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class EfficientAttention(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()

        if dim % heads != 0:
            raise ValueError("Embedding dimension must divide evenly by heads.")

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

        q = self.q(x)
        q = q.reshape(
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
            batch, reduced.shape[1], 2, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)

        output = attention @ v
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch, tokens, channels)

        return self.dropout(self.proj(output))


class MixFFN(nn.Module):
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

        x = x.transpose(1, 2).reshape(
            batch, hidden_channels, height, width
        )
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
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
        x = x + self.attention(self.norm1(x), height, width)
        x = x + self.mlp(self.norm2(x), height, width)
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
                SegFormerBlock(dims[i], heads[i], sr_ratios[i])
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
                        f"Expected square feature map; got {tokens} tokens."
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


model = SegFormerClassifier().to(DEVICE)

parameter_count = sum(p.numel() for p in model.parameters())
print("Parameters:", f"{parameter_count:,}")


# ============================================================
# METRICS AND EVALUATION
# ============================================================

def calculate_metrics(labels, probabilities, threshold=THRESHOLD):
    predictions = (probabilities >= threshold).astype(np.int64)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=EXPECTED_CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }

    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        metrics["roc_auc"] = None

    return metrics


def run_evaluation(model, loader, criterion=None, collect_paths=False):
    model.eval()

    all_labels = []
    all_probabilities = []
    all_paths = []
    total_loss = 0.0
    total_images = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(
            tqdm(loader, leave=False)
        ):
            images = images.to(DEVICE)
            labels_device = labels.float().to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):
                logits = model(images)
                loss = criterion(logits, labels_device) if criterion else None

            probabilities = torch.sigmoid(logits)

            batch_size = images.size(0)
            if loss is not None:
                total_loss += loss.item() * batch_size

            total_images += batch_size
            all_labels.extend(labels.numpy().astype(int).tolist())
            all_probabilities.extend(
                probabilities.float().cpu().numpy().tolist()
            )

            if collect_paths:
                start = batch_index * loader.batch_size
                end = start + batch_size
                all_paths.extend(
                    [path for path, _ in loader.dataset.samples[start:end]]
                )

    labels_array = np.asarray(all_labels, dtype=np.int64)
    probabilities_array = np.asarray(all_probabilities, dtype=np.float64)

    metrics = calculate_metrics(labels_array, probabilities_array)

    if criterion is not None and total_images:
        metrics["loss"] = float(total_loss / total_images)

    return metrics, labels_array, probabilities_array, all_paths


# ============================================================
# TRAINING
# ============================================================

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

best_val_f1 = -1.0
history = []
checkpoint_path = OUTPUT_DIR / "best_segformer.pt"

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()

    running_loss = 0.0
    processed = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )

    for images, labels in progress:
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        processed += batch_size

        progress.set_postfix(
            loss=f"{running_loss / max(processed, 1):.4f}"
        )

    train_loss = running_loss / max(processed, 1)
    val_metrics, _, _, _ = run_evaluation(model, val_loader, criterion)
    scheduler.step(val_metrics["macro_f1"])

    record = {
        "epoch": epoch + 1,
        "train_loss": float(train_loss),
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_precision": val_metrics["macro_precision"],
        "val_macro_recall": val_metrics["macro_recall"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "seconds": round(time.time() - epoch_start, 2),
    }
    history.append(record)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_accuracy={val_metrics['accuracy']:.4f} | "
        f"val_macro_F1={val_metrics['macro_f1']:.4f} | "
        f"val_ROC_AUC={val_metrics['roc_auc']}"
    )

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_macro_f1": best_val_f1,
                "classes": EXPECTED_CLASSES,
                "image_size": IMAGE_SIZE,
                "threshold": THRESHOLD,
            },
            checkpoint_path,
        )
        print("Saved best checkpoint:", checkpoint_path)

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2)


# ============================================================
# FINAL TEST: BEST VALIDATION CHECKPOINT
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state_dict"])

test_metrics, test_labels, test_probabilities, test_paths = run_evaluation(
    model,
    test_loader,
    collect_paths=True,
)

test_metrics["best_epoch"] = int(checkpoint["epoch"])
test_metrics["best_validation_macro_f1"] = float(
    checkpoint["best_val_macro_f1"]
)
test_metrics["classes"] = EXPECTED_CLASSES
test_metrics["image_size"] = IMAGE_SIZE
test_metrics["test_images"] = len(test_dataset)
test_metrics["dataset"] = "data/clean_dataset"

with open(
    OUTPUT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(test_metrics, file, indent=2)

# Confusion matrix image and CSV
cm = np.array(test_metrics["confusion_matrix"])

with open(
    OUTPUT_DIR / "confusion_matrix.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)
    writer.writerow(["Actual/Predicted"] + EXPECTED_CLASSES)
    for class_name, row in zip(EXPECTED_CLASSES, cm):
        writer.writerow([class_name] + row.tolist())

plt.figure(figsize=(6, 5))
plt.imshow(cm)