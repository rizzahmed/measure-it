"""Turn a Label Studio COCO export into one clean COCO file at full image size.

Label Studio was given half-size copies (2040 x 1148) of the undistorted photos.
This script scales every polygon by 2, so the masks match the real 4080 x 2296
images in dataset/images/, and gives each image its obj_XXX.jpg name.

Usage:
    python dataset/convert_export.py dataset/annotations/<export>.zip

Output: dataset/annotations/instances_full.json
Categories: 1 = book (top face of the cover), 2 = card.
"""
import argparse
import csv
import io
import json
import os
import re
import zipfile

import cv2
import numpy as np

SCALE = 2.0
FULL_W, FULL_H = 4080, 2296
HERE = os.path.dirname(os.path.abspath(__file__))


def load_export(path):
    with zipfile.ZipFile(path) as zf:
        return json.load(io.TextIOWrapper(zf.open("result.json"), encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_zip")
    parser.add_argument("--out", default=os.path.join(HERE, "annotations", "instances_full.json"))
    args = parser.parse_args()

    export = load_export(args.export_zip)
    names = {c["id"]: c["name"] for c in export["categories"]}
    card_present = {r["image"]: r["card_present"] == "1"
                    for r in csv.DictReader(open(os.path.join(HERE, "image_map.csv")))}

    image_name = {}
    for img in export["images"]:
        match = re.search(r"(obj_\d{3}\.jpg)$", img["file_name"])
        if not match:
            raise SystemExit(f"Cannot read image name from {img['file_name']}")
        image_name[img["id"]] = match.group(1)

    polygons = {}
    for ann in export["annotations"]:
        points = np.array(ann["segmentation"][0], float).reshape(-1, 2)
        if len(points) > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]  # COCO repeats the first point to close the polygon
        polygons.setdefault(image_name[ann["image_id"]], {})[names[ann["category_id"]]] = points * SCALE

    problems = []
    for name, has_card in sorted(card_present.items()):
        found = polygons.get(name, {})
        if "book" not in found:
            problems.append(f"{name}: no book polygon")
        if has_card != ("card" in found):
            problems.append(f"{name}: card_present={has_card} but card polygon found={'card' in found}")
    if problems:
        raise SystemExit("Labels do not match image_map.csv:\n" + "\n".join(problems))

    result = {"categories": [{"id": 1, "name": "book"}, {"id": 2, "name": "card"}],
              "images": [], "annotations": []}
    image_id = {}
    for i, name in enumerate(sorted(polygons), start=1):
        image_id[name] = i
        result["images"].append({"id": i, "file_name": name, "width": FULL_W, "height": FULL_H})
    ann_id = 1
    for name in sorted(polygons):
        for label, points in polygons[name].items():
            xs, ys = points[:, 0], points[:, 1]
            result["annotations"].append({
                "id": ann_id, "image_id": image_id[name],
                "category_id": 1 if label == "book" else 2,
                "segmentation": [[round(float(v), 1) for v in points.reshape(-1)]],
                "area": round(float(cv2.contourArea(points.astype(np.float32))), 1),
                "bbox": [round(float(xs.min()), 1), round(float(ys.min()), 1),
                         round(float(xs.max() - xs.min()), 1), round(float(ys.max() - ys.min()), 1)],
                "iscrowd": 0})
            ann_id += 1

    with open(args.out, "w") as fh:
        json.dump(result, fh)
    print(f"{len(result['images'])} images, {len(result['annotations'])} polygons -> {args.out}")


if __name__ == "__main__":
    main()
