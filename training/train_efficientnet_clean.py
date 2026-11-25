import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import random
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
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "clean_dataset"
OUTPUT_DIR = ROOT / "results" / "EfficientNetB0_Clean_Training"

CLASS_NAMES = ["damage", "no_damage"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("EfficientNetB0 — Clean Dataset Training")
print("=" * 65)
print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("Dataset:", DATA_DIR)
print("Output:", OUTPUT_DIR)


# ============================================================
# CHECK DATASET
# ============================================================

for split in ["train", "validation", "test"]:
    split_path = DATA_DIR / split
    if not split_path.exists():
        raise FileNotFoundError(f"Missing dataset folder: {split_path}")

    for class_name in CLASS_NAMES:
        class_path = split_path / class_name
        if not class_path.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_path}"
            )

        count = sum(
            1 for p in class_path.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        )
        print(f"{split:10s} | {class_name:10s} | {count:5d} images")


# ============================================================
# DATA PIPELINE
# Labels: damage = 0, no_damage = 1
#
# EfficientNetB0 includes its own input rescaling layers.
# Therefore, input images remain in the 0–255 range.
# ============================================================

def make_dataset(split, shuffle=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        class_names=CLASS_NAMES,
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    # Explicitly preserve raw pixel values (0–255).
    ds = ds.map(
        lambda images, labels: (
            tf.cast(images, tf.float32),
            tf.cast(labels, tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset("train", shuffle=True)
val_ds = make_dataset("validation", shuffle=False)
test_ds = make_dataset("test", shuffle=False)


# ============================================================
# DATA AUGMENTATION — TRAINING ONLY
# ============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="training_augmentation",
)


# ============================================================
# BUILD MODEL FROM IMAGENET-PRETRAINED EFFICIENTNETB0
# Fresh classification head; backbone frozen for fair transfer
# learning. The test set is not used for model selection.
# ============================================================

print("\nLoading ImageNet-pretrained EfficientNetB0...")
print("The first run may download pretrained weights.")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(128, 128, 3),
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(128, 128, 3), name="image")
x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)
x = tf.keras.layers.Dropout(0.30, name="dropout")(x)
outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    dtype="float32",
    name="damage_probability",
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="EfficientNetB0_Clean_BuildingDamage",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()


# ============================================================
# TRAINING
# ============================================================

best_model_path = OUTPUT_DIR / "best_efficientnetb0_clean.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(OUTPUT_DIR / "training_log.csv")
    ),
]

print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

with open(OUTPUT_DIR / "training_history.json", "w") as f:
    json.dump(
        {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
        f,
        indent=2,
    )


# ============================================================
# TRAINING CURVES
# ============================================================

hist = history.history

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(hist.get("accuracy", []), label="Training accuracy")
axes[0].plot(
    hist.get("val_accuracy", []),
    label="Validation accuracy",
)
axes[0].set_title("EfficientNetB0 Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hist.get("loss", []), label="Training loss")
axes[1].plot(
    hist.get("val_loss", []),
    label="Validation loss",
)
axes[1].set_title("EfficientNetB0 Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Binary cross-entropy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "training_curves.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\nLoading the best validation-loss checkpoint...")
best_model = tf.keras.models.load_model(
    best_model_path,
    compile=False,
)


# ============================================================
# SELECT CLASSIFICATION THRESHOLD USING VALIDATION ONLY
# Positive class is no_damage (label 1).
# Macro-F1 is calculated over both classes.
# ============================================================

print("\nSelecting threshold using validation data...")

y_val_parts = []
p_val_parts = []

for images, labels in val_ds:
    probabilities = best_model.predict(images, verbose=0).reshape(-1)
    y_val_parts.append(labels.numpy().reshape(-1).astype(int))
    p_val_parts.append(probabilities)

y_val = np.concatenate(y_val_parts)
p_val = np.concatenate(p_val_parts)

threshold_rows = []

for threshold in np.arange(0.10, 0.901, 0.01):
    predictions = (p_val >= threshold).astype(int)
    score = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0,
    )
    threshold_rows.append(
        {
            "threshold": round(float(threshold), 2),
            "validation_macro_f1": float(score),
        }
    )

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(
    OUTPUT_DIR / "validation_threshold_search.csv",
    index=False,
)

best_row = threshold_df.loc[
    threshold_df["validation_macro_f1"].idxmax()
]
selected_threshold = float(best_row["threshold"])

with open(OUTPUT_DIR / "selected_threshold.json", "w") as f:
    json.dump(
        {
            "threshold": selected_threshold,
            "selection_split": "validation",
            "selection_metric": "macro_f1",
            "validation_macro_f1": float(
                best_row["validation_macro_f1"]
            ),
        },
        f,
        indent=2,
    )

print(f"Selected threshold: {selected_threshold:.2f}")
print(
    "Validation macro-F1 at selected threshold:",
    round(float(best_row["validation_macro_f1"]), 4),
)


# ============================================================
# FINAL TEST EVALUATION — RUN ONCE AFTER THRESHOLD SELECTION
# ============================================================

print("\nEvaluating on the held-out test set...")

y_parts = []
p_parts = []
path_parts = []

# The dataset iterator provides images and labels in directory order,
# but not file paths. Reconstruct the same deterministic ordering.
for class_index, class_name in enumerate(CLASS_NAMES):
    class_dir = DATA_DIR / "test" / class_name
    files = sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    )
    path_parts.extend([(str(p), class_index) for p in files])

for images, labels in test_ds:
    probabilities = best_model.predict(images, verbose=0).reshape(-1)
    y_parts.append(labels.numpy().reshape(-1).astype(int))
    p_parts.append(probabilities)

y_true = np.concatenate(y_parts)
y_prob = np.concatenate(p_parts)

if len(y_true) != len(path_parts):
    raise RuntimeError(
        f"Test image count mismatch: labels={len(y_true)}, "
        f"paths={len(path_parts)}"
    )

# Confirm directory ordering corresponds to dataset labels.
path_labels = np.array([label for _, label in path_parts])
if not np.array_equal(y_true, path_labels):
    raise RuntimeError(
        "Test path/label order mismatch. Evaluation stopped to avoid "
        "saving incorrect image-level predictions."
    )

y_pred = (y_prob >= selected_threshold).astype(int)

metrics = {
    "model": "EfficientNetB0",
    "dataset": "clean_dataset",
    "class_order": CLASS_NAMES,
    "input_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
    "threshold": selected_threshold,
    "threshold_selection": "validation macro-F1",
    "test_samples": int(len(y_true)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_precision": float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    ),
    "macro_recall": float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    ),
    "macro_f1": float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    ),
    "weighted_f1": float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_true, y_prob)),
    "test_errors": int(np.sum(y_true != y_pred)),
}

with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

with open(OUTPUT_DIR / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
pd.DataFrame(
    cm,
    index=[f"actual_{c}" for c in CLASS_NAMES],
    columns=[f"predicted_{c}" for c in CLASS_NAMES],
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(cm)
ax.set_title("EfficientNetB0 — Clean Test Confusion Matrix")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_xticks([0, 1], CLASS_NAMES)
ax.set_yticks([0, 1], CLASS_NAMES)

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center", fontsize=12,
        )

fig.colorbar(image, ax=ax)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("EfficientNetB0 — Clean Test ROC Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# WRONG PREDICTIONS
# ============================================================

wrong_rows = []

for i, (image_path, true_label) in enumerate(path_parts):
    predicted_label = int(y_pred[i])
    if predicted_label != int(true_label):
        wrong_rows.append(
            {
                "image_path": image_path,
                "actual_class": CLASS_NAMES[int(true_label)],
                "predicted_class": CLASS_NAMES[predicted_label],
                "predicted_no_damage_probability": float(y_prob[i]),
                "threshold": selected_threshold,
            }
        )

pd.DataFrame(
    wrong_rows,
    columns=[
        "image_path",
        "actual_class",
        "predicted_class",
        "predicted_no_damage_probability",
        "threshold",
    ],
).to_csv(
    OUTPUT_DIR / "wrongly_predicted_test_images.csv",
    index=False,
)

print("\n" + "=" * 65)
print("CLEAN TEST RESULTS")
print("=" * 65)

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(cm)

print("\nWrongly predicted images:", len(wrong_rows))
print("All outputs saved to:", OUTPUT_DIR.resolve())
print("Training and evaluation completed.")
