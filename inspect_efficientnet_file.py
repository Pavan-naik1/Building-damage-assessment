
import zipfile
import json
import os

path = r"models\best_efficientnetb0.keras"

print("File exists:", os.path.exists(path))

if not os.path.exists(path):
    raise SystemExit("Model file not found.")

print("File size (MB):", round(os.path.getsize(path) / (1024 * 1024), 2))

with zipfile.ZipFile(path, "r") as z:
    print("\n===== FILE CONTENTS =====")
    for name in z.namelist():
        info = z.getinfo(name)
        print(name, "-", info.file_size, "bytes")

    if "config.json" in z.namelist():
        config = json.loads(z.read("config.json"))

        print("\n===== MODEL CONFIGURATION =====")
        print("Model class:", config.get("class_name"))

        model_config = config.get("config", {})
        for layer in model_config.get("layers", []):
            cls = layer.get("class_name")
            cfg = layer.get("config", {})
            print(
                "Layer:", cfg.get("name"),
                "| Type:", cls,
                "| Trainable:", cfg.get("trainable"),
                "| Units:", cfg.get("units"),
                "| Filters:", cfg.get("filters")
            )
