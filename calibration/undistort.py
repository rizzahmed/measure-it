"""Undistort photos with the saved calibration.

Usage:
    python calibration/undistort.py photo.jpg [more.jpg ...] --out calibration/undistorted

Straight edges (table edge, book, ruler) should look straight in the output.
Photos must be landscape (EXIF rotation applied) and the same size as the
calibration photos.
"""
import argparse
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_params(path=os.path.join(HERE, "params.npz")):
    data = np.load(path)
    return data["K"], data["dist"], tuple(int(v) for v in data["image_size"])


def undistort_image(img, K, dist):
    """Remove lens distortion while keeping the same size and camera matrix K,
    so pixel scale stays comparable between raw and undistorted images."""
    return cv2.undistort(img, K, dist, None, K)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--params", default=os.path.join(HERE, "params.npz"))
    parser.add_argument("--out", default=os.path.join(HERE, "undistorted"))
    args = parser.parse_args()

    K, dist, size = load_params(args.params)
    os.makedirs(args.out, exist_ok=True)
    for path in args.images:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # EXIF rotation applied
        if img is None:
            print(f"SKIP {path}: could not read")
            continue
        h, w = img.shape[:2]
        if (w, h) != size:
            print(f"SKIP {path}: size {w}x{h} differs from calibration {size[0]}x{size[1]}")
            continue
        out_path = os.path.join(args.out, os.path.basename(path))
        cv2.imwrite(out_path, undistort_image(img, K, dist))
        print(f"OK   {out_path}")


if __name__ == "__main__":
    main()
