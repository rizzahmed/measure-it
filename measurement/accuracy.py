"""Compare system measurements with the ruler values on a folder of new photos.

Usage (from the repository root):
    python measurement/accuracy.py measurement/photos --width 137 --height 210

Each photo shows the same book with the card, taken on a hard flat table, none of them
used for training. Runs the system on every photo twice (undistorted and raw) and writes
measurement/results.csv plus a printed summary. Refused photos are counted, not hidden.
"""
import argparse
import csv
import glob
import os
import statistics
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "inference"))
import measure as m  # noqa: E402


def summarise(label, rows, width, height):
    ok = [r for r in rows if r[f"{label}_status"] == "ok"]
    print(f"\n[{label}] measured {len(ok)} of {len(rows)} photos, refused {len(rows) - len(ok)}")
    if not ok:
        return
    for key, truth in (("width", width), ("height", height)):
        errs = [r[f"{label}_{key}"] - truth for r in ok]
        absolute = [abs(e) for e in errs]
        print(f"  {key}: ruler {truth} mm | mean error {statistics.mean(errs):+.1f} mm | "
              f"MAE {statistics.mean(absolute):.1f} mm ({100 * statistics.mean(absolute) / truth:.1f} %) | "
              f"median abs {statistics.median(absolute):.1f} mm | worst {max(absolute):.1f} mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--width", type=float, required=True, help="ruler width of the book cover in mm")
    parser.add_argument("--height", type=float, required=True, help="ruler height of the book cover in mm")
    parser.add_argument("--csv", default=os.path.join(HERE, "results.csv"), help="where to write the table")
    parser.add_argument("--run", default=os.path.join(ROOT, "models", "runs", "default"))
    args = parser.parse_args()

    cfg = m.load_config(os.path.join(ROOT, "models", "config.json"))
    K, dist, size = m.load_params()
    model = m.load_model(args.run, cfg)
    paths = sorted(p for p in glob.glob(os.path.join(args.folder, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    rows = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        row = {"image": os.path.basename(path)}
        if bgr is None or (bgr.shape[1], bgr.shape[0]) != size:
            row.update({"undistorted_status": "refused: unreadable or wrong size", "raw_status": "refused: unreadable or wrong size"})
            rows.append(row)
            continue
        for label, img in (("undistorted", m.undistort_image(bgr, K, dist)), ("raw", bgr)):
            result = m.measure(m.detect(model, img, cfg))
            info = result if isinstance(result, dict) else result[0]
            row[f"{label}_status"] = info["status"] if info["status"] == "ok" else f"refused: {info['reason']}"
            if info["status"] == "ok":
                row[f"{label}_width"], row[f"{label}_height"] = info["width_mm"], info["height_mm"]
        rows.append(row)
        print(row)

    fields = ["image", "undistorted_status", "undistorted_width", "undistorted_height",
              "raw_status", "raw_width", "raw_height"]
    with open(args.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for label in ("undistorted", "raw"):
        for r in rows:
            r[f"{label}_status"] = "ok" if r.get(f"{label}_status") == "ok" else "refused"
        summarise(label, rows, args.width, args.height)


if __name__ == "__main__":
    main()
