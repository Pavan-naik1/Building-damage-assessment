import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "clean_dataset"
TEST_DIR = DATA_DIR / "test"

MODEL_PATH = BASE_DIR / "models" / "best_cnn_optimized.keras"
RESULTS_DIR = BASE_DIR / "results" / "CNN_Clean"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
THRESHOLD = 0.50

CLASS_NAMES = ["damage", "no_damage"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD KERAS MODEL WITH COMPATIBILITY FIX
# The original model archive is not modified.
# ============================================================

print("=" * 65)
print("CNN CLEAN-TEST EVALUATION")
print("=" * 65)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"CNN model not found: {MODEL_PATH}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Clean test directory not found: {TEST_DIR}")

print("\nLoading CNN model...")
print("Model:", MODEL_PATH)

with tempfile.TemporaryDirectory() as temp_dir:
    compatible_model_path = Path(temp_dir) / "cnn_compatible.keras"

    with zipfile.ZipFile(MODEL_PATH, "r") as source_zip:
        with zipfile.ZipFile(
            compatible_model_path, "w", zipfile.ZIP_DEFLATED
        ) as target_zip:

            for item in source_zip.infolist():
                content = source_zip.read(item.filename)

                if item.filename == "config.json":
                    config = json.loads(content.decode("utf-8"))

                    def remove_null_quantization_config(obj):
                        if isinstance(obj, dict):
                            if obj.get("quantization_config", "missing") is None:
                                obj.pop("quantization_config")

                            for value in list(obj.values()):
                                remove_null_quantization_config(value)

                        elif isinstance(obj, list):
                            for value in obj:
                                remove_null_quantization_config(value)

                    remove_null_quantization_config(config)

                    content = json.dumps(config).encode("utf-8")

                target_zip.writestr(item, content)

    model = tf.keras.models.load_model(
        compatible_model_path,
        compile=False,
    )

print("CNN model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# ============================================================
# LOAD CLEAN TEST IMAGES
# Scaling is 1/255, as required for this CNN.
# ============================================================

print("\nLoading clean test images...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Detected class order:", test_dataset.class_names)

# Scale pixel values from [0, 255] to [0, 1].
test_dataset = test_dataset.map(
    lambda images, labels: (
        tf.cast(images, tf.float32) / 255.0,
        labels,
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true_batches = []
y_prob_batches = []

for images, labels in test_dataset:
    probabilities = model.predict(images, verbose=0).reshape(-1)

    y_true_batches.extend(labels.numpy().reshape(-1).astype(int))
    y_prob_batches.extend(probabilities.astype(float))

y_true = np.asarray(y_true_batches, dtype=int)
y_prob = np.asarray(y_prob_batches, dtype=float)

if len(y_true) == 0:
    raise ValueError("No test images were loaded.")

if not np.isfinite(y_prob).all():
    raise ValueError("Model predictions contain NaN or infinite values.")

if y_prob.min() < 0 or y_prob.max() > 1:
    raise ValueError(
        "CNN output is not in the expected probability range [0, 1]. "
        "Check whether the model output uses a sigmoid activation."
    )

y_pred = (y_prob >= THRESHOLD).astype(int)

# ============================================================
# METRICS
# Class 0 = damage; Class 1 = no_damage
# ============================================================

accuracy = accuracy_score(y_true, y_pred)
macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)

if len(np.unique(y_true)) == 2:
    roc_auc = roc_auc_score(y_true, y_prob)
else:
    roc_auc = None

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

metrics = {
    "model": "CNN",
    "dataset": "clean_dataset/test",
    "test_samples": int(len(y_true)),
    "class_order": CLASS_NAMES,
    "positive_class_for_auc": "no_damage (class 1)",
    "threshold": float(THRESHOLD),
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "roc_auc": float(roc_auc) if roc_auc is not None else None,
    "damage_precision": float(report["damage"]["precision"]),
    "damage_recall": float(report["damage"]["recall"]),
    "damage_f1": float(report["damage"]["f1-score"]),
    "no_damage_precision": float(report["no_damage"]["precision"]),
    "no_damage_recall": float(report["no_damage"]["recall"]),
    "no_damage_f1": float(report["no_damage"]["f1-score"]),
    "confusion_matrix_labels": CLASS_NAMES,
    "confusion_matrix": cm.tolist(),
}

with open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

with open(RESULTS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

pd.DataFrame(
    cm,
    index=[f"Actual_{name}" for name in CLASS_NAMES],
    columns=[f"Predicted_{name}" for name in CLASS_NAMES],
).to_csv(RESULTS_DIR / "confusion_matrix.csv")

# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))
image = ax.imshow(cm, interpolation="nearest")
ax.figure.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(len(CLASS_NAMES)),
    yticks=np.arange(len(CLASS_NAMES)),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted label",
    ylabel="True label",
    title=f"CNN Confusion Matrix (threshold={THRESHOLD:.2f})",
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=12,
        )

fig.tight_layout()
fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

# ============================================================
# ROC CURVE
# ============================================================

if roc_auc is not None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"CNN (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="CNN ROC Curve — Clean Test Set",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

# ============================================================
# WRONGLY PREDICTED IMAGES
# ============================================================

print("\nRecording misclassified test images...")

image_paths = []
image_labels = []

for class_name in CLASS_NAMES:
    class_dir = TEST_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class folder: {class_dir}")

    files = sorted(
        path for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    image_paths.extend(files)
    image_labels.extend([CLASS_NAMES.index(class_name)] * len(files))

if len(image_paths) != len(y_true):
    raise RuntimeError(
        f"Image/prediction count mismatch: {len(image_paths)} image paths "
        f"but {len(y_true)} predictions. Check the test directory contents."
    )

wrong_indices = np.flatnonzero(y_true != y_pred)

wrong_rows = []

for idx in wrong_indices:
    wrong_rows.append({
        "image_path": str(image_paths[idx].resolve()),
        "actual_class": CLASS_NAMES[int(y_true[idx])],
        "predicted_class": CLASS_NAMES[int(y_pred[idx])],
        "predicted_no_damage_probability": float(y_prob[idx]),
        "threshold": float(THRESHOLD),
    })

pd.DataFrame(
    wrong_rows,
    columns=[
        "image_path",
        "actual_class",
        "predicted_class",
        "predicted_no_damage_probability",
        "threshold",
    ],
).to_csv(RESULTS_DIR / "wrongly_predicted_test_images.csv", index=False)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CNN CLEAN TEST RESULTS")
print("=" * 65)

print(f"Test samples       : {len(y_true)}")
print(f"Threshold          : {THRESHOLD:.2f}")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Macro precision    : {macro_precision:.4f}")
print(f"Macro recall       : {macro_recall:.4f}")
print(f"Macro F1-score     : {macro_f1:.4f}")
print(f"Weighted F1-score  : {weighted_f1:.4f}")
print(f"ROC-AUC            : {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC            : N/A")

print("\nPer-class report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix (rows=actual, columns=predicted):")
print(cm)

print(f"\nMisclassified images: {len(wrong_indices)}")
print("Results saved to:", RESULTS_DIR)
print("\nEvaluation completed successfully.")

