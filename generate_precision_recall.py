
from pathlib import Path
import json
import gc
import csv
import zipfile
import tempfile
import shutil

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    recall_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data" / "clean_dataset" / "test"

CNN_PATH = ROOT / "results" / "CNN_Clean_Training" / "best_cnn_clean.keras"
EFF_PATH = ROOT / "results" / "EfficientNetB0_Clean_Training" / "best_efficientnetb0_clean.keras"
SEG_PATH = ROOT / "results" / "SegFormer_Clean" / "best_segformer.pt"

OUTPUT_DIR = ROOT / "results" / "Precision_Recall_Analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["damage", "no_damage"]
BATCH_SIZE = 32

# Thresholds from the original 10-epoch evaluation.
THRESHOLDS = {
    "CNN": 0.50,
    "EfficientNetB0": 0.55,
}

# ============================================================
# COMMON HELPERS
# ============================================================

def check_files():
    required = [TEST_DIR, CNN_PATH, EFF_PATH, SEG_PATH]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    print("All dataset and checkpoint paths verified.")


def calculate_metrics(name, labels, p_no_damage, threshold, paths):
    labels = np.asarray(labels, dtype=np.int64)
    p_no_damage = np.asarray(p_no_damage, dtype=np.float64).reshape(-1)

    if len(labels) != len(p_no_damage):
        raise ValueError(f"{name}: label/probability count mismatch.")

    if len(paths) != len(labels):
        raise ValueError(f"{name}: image path count mismatch.")

    # Dataset class order: damage=0, no_damage=1.
    y_damage = (labels == 0).astype(int)
    y_no_damage = (labels == 1).astype(int)

    p_damage = 1.0 - p_no_damage

    # Existing decision rule:
    # p(no_damage) >= threshold -> no_damage
    # otherwise -> damage
    predictions = (p_no_damage >= threshold).astype(int)

    precision_damage, recall_damage, _ = precision_recall_curve(
        y_damage, p_damage
    )

    precision_no_damage, recall_no_damage, _ = precision_recall_curve(
        y_no_damage, p_no_damage
    )

    ap_damage = average_precision_score(y_damage, p_damage)
    ap_no_damage = average_precision_score(y_no_damage, p_no_damage)

    damage_recall = recall_score(
        y_damage,
        (p_no_damage < threshold).astype(int),
        zero_division=0,
    )

    no_damage_recall = recall_score(
        y_no_damage,
        (p_no_damage >= threshold).astype(int),
        zero_division=0,
    )

    metrics = {
        "model": name,
        "test_samples": int(len(labels)),
        "threshold_on_no_damage_probability": float(threshold),
        "damage_average_precision": float(ap_damage),
        "no_damage_average_precision": float(ap_no_damage),
        "damage_recall_at_threshold": float(damage_recall),
        "no_damage_recall_at_threshold": float(no_damage_recall),
    }

    predictions_data = []

    for i in range(len(labels)):
        predictions_data.append({
            "model": name,
            "image_path": str(paths[i]),
            "actual_class": CLASS_NAMES[labels[i]],
            "predicted_class": CLASS_NAMES[predictions[i]],
            "p_damage": float(p_damage[i]),
            "p_no_damage": float(p_no_damage[i]),
            "threshold": float(threshold),
        })

    curves = {
        "damage": (precision_damage, recall_damage, ap_damage),
        "no_damage": (precision_no_damage, recall_no_damage, ap_no_damage),
    }

    print(f"\n{name} results")
    print("-" * 45)
    print(f"Test samples: {len(labels)}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Damage Average Precision: {ap_damage:.4f}")
    print(f"No-damage Average Precision: {ap_no_damage:.4f}")
    print(f"Damage Recall: {damage_recall:.4f}")
    print(f"No-damage Recall: {no_damage_recall:.4f}")

    return metrics, predictions_data, curves


# ============================================================
# TENSORFLOW MODEL LOADING COMPATIBILITY
# ============================================================

def load_keras_model_compatible(model_path):
    import tensorflow as tf

    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
        )
    except Exception as first_error:
        print("Trying compatibility loading for Keras checkpoint.")
        print("Initial loading message:", first_error)

    # Make a temporary copy and remove null quantization_config
    # fields if present. The original checkpoint is never changed.
    temp_dir = Path(tempfile.mkdtemp(prefix="keras_pr_"))
    temp_model = temp_dir / model_path.name

    try:
        with zipfile.ZipFile(model_path, "r") as source:
            with zipfile.ZipFile(temp_model, "w") as target:
                for item in source.infolist():
                    content = source.read(item.filename)

                    if item.filename == "config.json":
                        config = json.loads(content.decode("utf-8"))

                        def clean_config(obj):
                            if isinstance(obj, dict):
                                if obj.get("quantization_config") is None:
                                    obj.pop("quantization_config", None)

                                for value in obj.values():
                                    clean_config(value)

                            elif isinstance(obj, list):
                                for value in obj:
                                    clean_config(value)

                        clean_config(config)
                        content = json.dumps(config).encode("utf-8")

                    target.writestr(item, content)

        model = tf.keras.models.load_model(
            temp_model,
            compile=False,
        )

        return model

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# CNN
# ============================================================

def evaluate_cnn():
    print("\n" + "=" * 65)
    print("EVALUATING ORIGINAL CNN CHECKPOINT")
    print("=" * 65)

    import tensorflow as tf

    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(128, 128),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    paths = list(dataset.file_paths)

    dataset = dataset.map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels,
        )
    )

    model = load_keras_model_compatible(CNN_PATH)

    probabilities = model.predict(dataset, verbose=1).reshape(-1)

    labels = np.concatenate([
        labels.numpy() for _, labels in dataset
    ])

    result = calculate_metrics(
        "CNN",
        labels,
        probabilities,
        THRESHOLDS["CNN"],
        paths,
    )

    del model, dataset
    tf.keras.backend.clear_session()
    gc.collect()

    return result


# ============================================================
# EFFICIENTNETB0
# ============================================================

def evaluate_efficientnet():
    print("\n" + "=" * 65)
    print("EVALUATING ORIGINAL EFFICIENTNETB0 CHECKPOINT")
    print("=" * 65)

    import tensorflow as tf

    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(128, 128),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    paths = list(dataset.file_paths)

    # Preserve the preprocessing used in the original
    # EfficientNet evaluation: do not divide by 255 here.
    model = load_keras_model_compatible(EFF_PATH)

    probabilities = model.predict(dataset, verbose=1).reshape(-1)

    labels = np.concatenate([
        labels.numpy() for _, labels in dataset
    ])

    result = calculate_metrics(
        "EfficientNetB0",
        labels,
        probabilities,
        THRESHOLDS["EfficientNetB0"],
        paths,
    )

    del model, dataset
    tf.keras.backend.clear_session()
    gc.collect()

    return result


# ============================================================
# SEGFORMER
# ============================================================

def evaluate_segformer():
    print("\n" + "=" * 65)
    print("EVALUATING ORIGINAL SEGFORMER CHECKPOINT")
    print("=" * 65)

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from evaluate_segformer_corrected import (
        SegFormerClassifier,
        DEVICE,
    )

    checkpoint = torch.load(
        SEG_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    expected_classes = checkpoint.get(
        "classes", CLASS_NAMES
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
        str(TEST_DIR),
        transform=transform,
    )

    if test_dataset.classes != expected_classes:
        raise ValueError(
            f"Class mismatch: {test_dataset.classes} vs {expected_classes}"
        )

    loader = DataLoader(
        test_dataset,
        batch_size=8,
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

    labels = []
    probabilities = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(DEVICE)

            logits = model(images)
            p_no_damage = torch.sigmoid(logits)

            labels.extend(batch_labels.numpy().tolist())
            probabilities.extend(
                p_no_damage.cpu().numpy().reshape(-1).tolist()
            )

    paths = [sample[0] for sample in test_dataset.samples]

    result = calculate_metrics(
        "SegFormer",
        labels,
        probabilities,
        threshold,
        paths,
    )

    del model, loader, test_dataset, checkpoint
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return result


# ============================================================
# SAVE RESULTS AND PLOT
# ============================================================

def save_outputs(results):
    metrics_list = [item[0] for item in results]
    predictions_list = [item[1] for item in results]
    curves_list = [item[2] for item in results]

    with open(
        OUTPUT_DIR / "precision_recall_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics_list, file, indent=4)

    csv_path = OUTPUT_DIR / "all_model_test_predictions.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "model",
            "image_path",
            "actual_class",
            "predicted_class",
            "p_damage",
            "p_no_damage",
            "threshold",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for model_predictions in predictions_list:
            writer.writerows(model_predictions)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for name, curves in zip(
        ["CNN", "EfficientNetB0", "SegFormer"],
        curves_list,
    ):
        precision, recall, ap = curves["damage"]

        axes[0].plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.3f})",
        )

        precision, recall, ap = curves["no_damage"]

        axes[1].plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.3f})",
        )

    axes[0].set_title("Precision-Recall Curve: Damage")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Precision-Recall Curve: No Damage")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        "Precision-Recall Comparison of Three Classifiers",
        fontsize=13,
    )

    fig.tight_layout()

    figure_path = OUTPUT_DIR / "precision_recall_comparison.png"

    fig.savefig(
        figure_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n" + "=" * 65)
    print("PRECISION-RECALL ANALYSIS COMPLETED")
    print("=" * 65)
    print("Output directory:", OUTPUT_DIR)
    print("Figure:", figure_path)
    print("Metrics:", OUTPUT_DIR / "precision_recall_metrics.json")
    print("Predictions:", csv_path)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("PRECISION-RECALL ANALYSIS")
    print("Original 10-epoch checkpoints only")
    print("=" * 65)

    check_files()

    results = []

    results.append(evaluate_cnn())
    results.append(evaluate_efficientnet())
    results.append(evaluate_segformer())

    save_outputs(results)


if __name__ == "__main__":
    main()
