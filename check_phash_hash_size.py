from pathlib import Path
import inspect
import imagehash

file_path = Path("analyze_phash_duplicates.py")

print("========== pHash HASH SIZE CHECK ==========")

if file_path.exists():
    lines = file_path.read_text(encoding="utf-8").splitlines()

    print("\nActual pHash calls:")

    for number, line in enumerate(lines, start=1):
        if "imagehash.phash" in line:
            print(f"Line {number}: {line.strip()}")
else:
    print("ERROR: analyze_phash_duplicates.py not found.")

signature = inspect.signature(imagehash.phash)
default_size = signature.parameters["hash_size"].default

print("\nDefault hash size:", default_size)
print("Hash dimensions:", f"{default_size} x {default_size}")
print("Total hash bits:", default_size * default_size)

print("\n============================================")
