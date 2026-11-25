
from pathlib import Path
import csv
import math

INPUT = Path("results/Geographic_Analysis/coordinate_audit.csv")
OUTPUT = Path("results/Geographic_Analysis/nearby_cross_split_pairs_50m.csv")

DISTANCE_THRESHOLD_METRES = 50


def haversine(lat1, lon1, lat2, lon2):
    earth_radius = 6371000

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * math.asin(math.sqrt(a))


def main():
    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))

    # Avoid comparing an image with itself.
    pairs = []

    print("Images loaded:", len(records))
    print("Checking geographic distances...")

    for i in range(len(records)):
        a = records[i]

        lat1 = float(a["latitude"])
        lon1 = float(a["longitude"])

        for j in range(i + 1, len(records)):
            b = records[j]

            if a["split"] == b["split"]:
                continue

            lat2 = float(b["latitude"])
            lon2 = float(b["longitude"])

            distance = haversine(lat1, lon1, lat2, lon2)

            if distance <= DISTANCE_THRESHOLD_METRES:
                pairs.append({
                    "image_1": a["source_path"],
                    "split_1": a["split"],
                    "class_1": a["class"],
                    "image_2": b["source_path"],
                    "split_2": b["split"],
                    "class_2": b["class"],
                    "distance_metres": round(distance, 2)
                })

    fields = [
        "image_1", "split_1", "class_1",
        "image_2", "split_2", "class_2",
        "distance_metres"
    ]

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)

    print("\nGEOGRAPHIC DISTANCE AUDIT")
    print("-------------------------")
    print("Distance threshold:", DISTANCE_THRESHOLD_METRES, "metres")
    print("Cross-split nearby pairs:", len(pairs))
    print("Report saved:", OUTPUT.resolve())
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
