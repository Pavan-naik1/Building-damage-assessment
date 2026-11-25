
import zipfile
import h5py
import math
import os
import tempfile

model_path = r"models\best_efficientnetb0.keras"

with tempfile.TemporaryDirectory() as temp_dir:
    with zipfile.ZipFile(model_path, "r") as archive:
        archive.extract("model.weights.h5", temp_dir)

    weights_path = os.path.join(temp_dir, "model.weights.h5")

    totals = {"all_model_weights": 0, "dense_head": 0}

    with h5py.File(weights_path, "r") as f:
        def inspect(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return

            # Ignore optimizer slots and variables
            if name.startswith("optimizer/"):
                return

            count = math.prod(obj.shape) if obj.shape else 1
            totals["all_model_weights"] += count

            if name.startswith("layers/dense/"):
                totals["dense_head"] += count

        f.visititems(inspect)

    print("Total model weight values:", f"{totals['all_model_weights']:,}")
    print("Dense classification head:", f"{totals['dense_head']:,}")
    print("Other model weight values:",
          f"{totals['all_model_weights'] - totals['dense_head']:,}")
