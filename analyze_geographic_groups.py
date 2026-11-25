
from pathlib import Path
import csv
import re
from collections import defaultdict

MANIFEST = Path("data/clean_dataset/manifest.csv")
OUTPUT_DIR = Path("results/Geographic_Analysis")

# Extract coordinates from filenames such as:
# -93.55964_30.895018.jpeg
COORD_PATTERN = re.compile(
    r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    coordinate_splits = defaultdict(set)
    invalid_files = []

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = Path(row["source_path"]).stem
            match = COORD_PATTERN.match(filename)

            if not match:
                invalid_files.append(row["source_path"])
                continue

            # Filename pattern: longitude_latitude
            longitude = float(match.group(1))
            latitude = float(match.group(2))

            # Basic geographic validity check
            if not (-180 <= longitude <= 180 and
                    -90 <= latitude <= 90):
                invalid_files.append(row["source_path"])
                continue

            coordinate_key = (
                round(longitude, 5),
                round(latitude, 5)
            )

            coordinate_splits[coordinate_key].add(row["split"])

            records.append({
                "longitude": longitude,
                "latitude": latitude,
                "split": row["split"],
                "class": row["class"],
                "source_path": row["source_path"],
                "coordinate_key": str(coordinate_key)
            })

    # Save coordinate extraction report
    with (OUTPUT_DIR / "coordinate_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        fields = [
            "longitude", "latitude", "split",
            "class", "source_path", "coordinate_key"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    # Find coordinates represented in multiple splits
    cross_split = []

    for key, splits in coordinate_splits.items():
        if len(splits) > 1:
            cross_split.append({
                "coordinate": str(key),
                "splits": ";".join(sorted(splits))
            })

    with (OUTPUT_DIR / "cross_split_coordinates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        fields = ["coordinate", "splits"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cross_split)

    print("\nGEOGRAPHIC COORDINATE AUDIT")
    print("---------------------------")
    print("Valid coordinate records:", len(records))
    print("Invalid/unparsed filenames:", len(invalid_files))
    print("Unique rounded coordinates:", len(coordinate_splits))
    print("Coordinates appearing in multiple splits:", len(cross_split))

    print("\nRecords by split:")
    for split in ["train", "validation", "test"]:
        count = sum(1 for row in records if row["split"] == split)
        print(f"{split}: {count}")

    print("\nReports saved in:", OUTPUT_DIR.resolve())
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
