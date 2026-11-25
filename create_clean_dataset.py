
from pathlib import Path
from collections import defaultdict
import hashlib
import random
import shutil
import csv

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
SOURCE = Path(
    r"C:\Users\Pavan Naik\OneDrive\Desktop\Minor Project\archive (12)"
)
DEST = Path("data") / "clean_dataset"

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train_another", "validation_another", "test"]
CLASSES = ["damage", "no_damage"]

# Stop rather than overwrite an existing dataset.
if DEST.exists():
    raise FileExistsError(
        f"{DEST} already exists. Rename or remove it only after checking its contents."
    )

random.seed(SEED)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    # Each exact file-content hash becomes one indivisible group.
    groups = defaultdict(list)
    hash_labels = defaultdict(set)
    hash_splits = defaultdict(set)

    for split in SPLITS:
        split_dir = SOURCE / split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)

        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(class_dir)

            for path in class_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                digest = sha256(path)
                groups[digest].append((split, class_name, path))
                hash_labels[digest].add(class_name)
                hash_splits[digest].add(split)

    # Do not proceed if identical bytes have different labels.
    conflicts = [
        digest for digest, labels in hash_labels.items()
        if len(labels) > 1
    ]
    if conflicts:
        raise ValueError(
            f"Found {len(conflicts)} duplicate-content groups with conflicting labels."
        )

    # Group by class; all identical copies remain in one split.
    groups_by_class = defaultdict(list)
    for digest, entries in groups.items():
        class_name = entries[0][1]
        groups_by_class[class_name].append((digest, entries))

    # Assign unique image groups to 70% train, 15% validation, 15% test.
    assignments = {}
    for class_name, class_groups in groups_by_class.items():
        random.shuffle(class_groups)
        n = len(class_groups)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        for i, (digest, entries) in enumerate(class_groups):
            if i < train_end:
                split_name = "train"
            elif i < val_end:
                split_name = "validation"
            else:
                split_name = "test"

            assignments[digest] = split_name

    # Create output folders and copy one representative per unique image.
    manifest_path = DEST / "manifest.csv"
    rows = []

    for split_name in ["train", "validation", "test"]:
        for class_name in CLASSES:
            (DEST / split_name / class_name).mkdir(parents=True, exist_ok=True)

    for digest, entries in groups.items():
        split_name = assignments[digest]
        class_name = entries[0][1]
        source_path = entries[0][2]

        # Preserve the original extension.
        filename = digest + source_path.suffix.lower()
        target_path = DEST / split_name / class_name / filename
        shutil.copy2(source_path, target_path)

        rows.append({
            "sha256": digest,
            "split": split_name,
            "class": class_name,
            "source_path": str(source_path),
            "output_path": str(target_path.resolve()),
            "source_copies": len(entries),
            "source_splits": ";".join(sorted(hash_splits[digest])),
        })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Clean dataset created:", DEST.resolve())
    print("Unique image contents:", len(groups))
    print("Conflicting-label groups:", len(conflicts))

    for split_name in ["train", "validation", "test"]:
        print(f"\n{split_name}:")
        for class_name in CLASSES:
            count = len(list((DEST / split_name / class_name).glob("*")))
            print(f"  {class_name}: {count}")

    print("\nManifest:", manifest_path.resolve())
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
