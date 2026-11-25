
from pathlib import Path
from collections import defaultdict
import hashlib

DATA_ROOT = Path(
    r"C:\Users\Pavan Naik\OneDrive\Desktop\Minor Project\archive (12)"
)

SPLITS = {
    "train": DATA_ROOT / "train_another",
    "validation": DATA_ROOT / "validation_another",
    "test": DATA_ROOT / "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    hashes_by_split = {}

    for split_name, split_path in SPLITS.items():
        if not split_path.is_dir():
            print(f"ERROR: Folder not found: {split_path}")
            return

        hashes = defaultdict(list)

        for path in split_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                hashes[image_hash(path)].append(path)

        hashes_by_split[split_name] = hashes
        image_count = sum(len(paths) for paths in hashes.values())
        unique_count = len(hashes)

        print(
            f"{split_name}: {image_count} images, "
            f"{unique_count} unique file contents"
        )

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for first, second in pairs:
        first_hashes = hashes_by_split[first]
        second_hashes = hashes_by_split[second]

        shared_hashes = set(first_hashes) & set(second_hashes)
        shared_first = sum(len(first_hashes[h]) for h in shared_hashes)
        shared_second = sum(len(second_hashes[h]) for h in shared_hashes)

        print(f"\n{first} vs {second}:")
        print(f"  Shared unique image contents: {len(shared_hashes)}")
        print(f"  Matching files in {first}: {shared_first}")
        print(f"  Matching files in {second}: {shared_second}")


if __name__ == "__main__":
    main()
