
from pathlib import Path
from PIL import Image
import imagehash
import csv

# Existing clean dataset (read-only)
DATASET = Path("data/clean_dataset")

SPLITS = ["train", "validation", "test"]
CLASSES = ["damage", "no_damage"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# pHash distance threshold:
# Smaller distance means images are more visually similar.
PHASH_THRESHOLD = 5

def main():
    images = []

    print("Reading images and calculating perceptual hashes...")

    for split in SPLITS:
        for class_name in CLASSES:
            folder = DATASET / split / class_name

            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    try:
                        with Image.open(path) as img:
                            phash = imagehash.phash(img.convert("RGB"))

                        images.append({
                            "path": str(path),
                            "split": split,
                            "class": class_name,
                            "hash": phash
                        })

                    except Exception as error:
                        print("Could not process:", path, error)

    print("Images processed:", len(images))
    print("Comparing perceptual hashes...")

    cross_split_matches = []
    same_split_matches = []

    # Compare only images from different files.
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            distance = images[i]["hash"] - images[j]["hash"]

            if distance <= PHASH_THRESHOLD:
                match = {
                    "image_1": images[i]["path"],
                    "split_1": images[i]["split"],
                    "class_1": images[i]["class"],
                    "image_2": images[j]["path"],
                    "split_2": images[j]["split"],
                    "class_2": images[j]["class"],
                    "phash_distance": distance
                }

                if images[i]["split"] != images[j]["split"]:
                    cross_split_matches.append(match)
                else:
                    same_split_matches.append(match)

    output = Path("results") / "pHash_Analysis"
    output.mkdir(parents=True, exist_ok=True)

    fields = [
        "image_1", "split_1", "class_1",
        "image_2", "split_2", "class_2",
        "phash_distance"
    ]

    with (output / "cross_split_matches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cross_split_matches)

    with (output / "same_split_matches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(same_split_matches)

    print("\nPHASH ANALYSIS COMPLETED")
    print("Images processed:", len(images))
    print("Cross-split similar pairs:", len(cross_split_matches))
    print("Same-split similar pairs:", len(same_split_matches))
    print("Threshold:", PHASH_THRESHOLD)
    print("Reports saved in:", output.resolve())
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
