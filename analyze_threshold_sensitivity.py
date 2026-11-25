
from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path(
    "results/Precision_Recall_Analysis/all_model_test_predictions.csv"
)

OUTPUT_DIR = Path("results/Threshold_Sensitivity_Analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds applied to P(damage)
THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.60, 0.70]

# Original thresholds converted from P(no_damage)
ORIGINAL_THRESHOLDS = {
    "CNN": 0.50,
    "EfficientNetB0": 0.45,
    "SegFormer": 0.50
}

# --------------------------------------------------
# LOAD PREDICTIONS
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {INPUT_FILE}"
    )

with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise ValueError("Prediction CSV is empty.")

models = sorted(set(row["model"] for row in rows))

results = []

# --------------------------------------------------
# THRESHOLD SENSITIVITY ANALYSIS
# --------------------------------------------------

for model_name in models:

    model_rows = [
        row for row in rows
        if row["model"] == model_name
    ]

    y_true = np.array([
        1 if row["actual_class"] == "damage" else 0
        for row in model_rows
    ])

    probabilities = np.array([
        float(row["p_damage"])
        for row in model_rows
    ])

    original_threshold = ORIGINAL_THRESHOLDS[model_name]

    print(f"\nAnalyzing: {model_name}")
    print("-" * 65)

    for threshold in THRESHOLDS:

        y_pred = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        result = {
            "model": model_name,
            "damage_probability_threshold": threshold,
            "is_original_operating_threshold": (
                abs(threshold - original_threshold) < 1e-9
            ),
            "test_samples": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "damage_precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "damage_recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "damage_f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }

        results.append(result)

        print(
            f"Threshold: {threshold:.2f} | "
            f"Damage Recall: {result['damage_recall']:.4f} | "
            f"Damage Precision: {result['damage_precision']:.4f} | "
            f"Missed Damage: {fn} | "
            f"False Alarms: {fp}"
        )

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

csv_path = OUTPUT_DIR / "threshold_sensitivity_results.csv"
json_path = OUTPUT_DIR / "threshold_sensitivity_results.json"

fieldnames = list(results[0].keys())

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

with json_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 65)
print("THRESHOLD SENSITIVITY ANALYSIS COMPLETED")
print("=" * 65)

print(f"Models analyzed: {len(models)}")
print(f"Thresholds tested: {THRESHOLDS}")
print(f"Total result rows: {len(results)}")
print(f"CSV saved: {csv_path}")
print(f"JSON saved: {json_path}")
