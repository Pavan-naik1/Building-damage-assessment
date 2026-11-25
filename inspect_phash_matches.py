
from pathlib import Path
from PIL import Image, ImageDraw
import csv
import math

csv_path = Path("results/pHash_Analysis/cross_split_matches.csv")
output_path = Path("results/pHash_Analysis/phash_contact_sheet.jpg")

with csv_path.open("r", encoding="utf-8") as f:
    matches = list(csv.DictReader(f))

thumb_w, thumb_h = 220, 180
label_h = 65
columns = 2
rows = len(matches)

sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows * (thumb_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, match in enumerate(matches):
    for side in ["1", "2"]:
        path = Path(match[f"image_{side}"])
        x = (i % columns) * thumb_w
        y = (i // columns) * (thumb_h + label_h)

        if side == "2":
            x += thumb_w // 2
            # Place the second image beside the first.
            # Use a wider pair layout below instead.

    # Each pair gets its own row with two images.
    pair_w = thumb_w * 2
    y = i * (thumb_h + label_h)

    for side, x in [("1", 0), ("2", thumb_w)]:
        path = Path(match[f"image_{side}"])

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb_w - 10, thumb_h - 10))
                px = x + (thumb_w - img.width) // 2
                py = y + (thumb_h - img.height) // 2
                sheet.paste(img, (px, py))

            draw.text(
                (x + 5, y + thumb_h),
                f"{match[f'split_{side}']} / {match[f'class_{side}']}",
                fill="black"
            )
        except Exception as error:
            draw.text((x + 5, y + 10), str(error), fill="red")

    draw.text(
        (5, y + thumb_h + 20),
        f"Pair {i+1} | pHash distance: {match['phash_distance']}",
        fill="blue"
    )

sheet.save(output_path, quality=90)
print("Contact sheet saved:", output_path.resolve())
