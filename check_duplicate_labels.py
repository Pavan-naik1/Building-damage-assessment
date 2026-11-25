
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(r"C:\Users\Pavan Naik\OneDrive\Desktop\Minor Project\archive (12)")
SPLITS = ["train_another", "validation_another", "test"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Hash -> list of (split, class, path)
records = defaultdict(list)

for split in SPLITS:
    folder = ROOT / split
    for class_dir in folder.iterdir():
        if not class_dir.is_dir():
            continue
        for path in class_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXTS:
                continue
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            records[h].append((split, class_dir.name, str(path)))

cross_split = [
    entries for entries in records.values()
    if len({e[0] for e in entries}) > 1
]

conflicts = [
    entries for entries in cross_split
    if len({e[1] for e in entries}) > 1
]

print("Unique image contents appearing across splits:", len(cross_split))
print("Cross-split duplicates with conflicting class labels:", len(conflicts))

for entries in conflicts[:10]:
    print("\nCONFLICT:")
    for split, label, path in entries:
        print(f"  {split} | {label} | {path}")
