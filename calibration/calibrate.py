"""Calibrate the camera from checkerboard photos and save the parameters.

Usage:
    python calibration/calibrate.py --images Chessboard2 Chessboard3 --square-mm 19.5

Outputs (next to this script):
    params.npz   camera matrix K, distortion coefficients, image size
    report.txt   detection status and reprojection error of every photo
    detected/    small previews with the detected corners drawn on them

Photos are read with their EXIF rotation applied. Photos that are still
portrait after that are skipped, because a calibration needs one fixed image
size. Use --rotate-portrait cw|ccw to include them anyway.
"""
import argparse
import glob
import os

import cv2
import numpy as np

# calib.io 9 x 11 squares  ->  10 x 8 inner corners (columns, rows)
PATTERN = (10, 8)
EXTENSIONS = ("jpg", "jpeg", "png")
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
SEARCH_SCALE = 0.25  # find the board on a small copy, refine on the full image
HERE = os.path.dirname(os.path.abspath(__file__))


def list_images(folders):
    paths = []
    for folder in folders:
        for ext in EXTENSIONS:
            paths.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
            paths.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))
    return sorted(set(paths))


def board_points(square_mm):
    """Real-world corner positions in mm on the flat board (Z = 0)."""
    pts = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    pts[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * square_mm
    return pts


def find_corners(gray):
    small = cv2.resize(gray, None, fx=SEARCH_SCALE, fy=SEARCH_SCALE, interpolation=cv2.INTER_AREA)
    found, corners = cv2.findChessboardCorners(
        small, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        found, corners = cv2.findChessboardCornersSB(small, PATTERN, cv2.CALIB_CB_ACCURACY)
    if not found:
        return None
    corners = (corners.reshape(-1, 1, 2) / SEARCH_SCALE).astype(np.float32)
    return cv2.cornerSubPix(gray, corners, (23, 23), (-1, -1), CRITERIA)


def rotate_points(pts, width, height, direction):
    """Map corners of a portrait image (width x height) into the rotated landscape image."""
    pts = pts.reshape(-1, 2)
    if direction == "cw":
        out = np.stack([height - 1 - pts[:, 1], pts[:, 0]], axis=1)
    else:
        out = np.stack([pts[:, 1], width - 1 - pts[:, 0]], axis=1)
    return out.reshape(-1, 1, 2).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, help="one or more photo folders")
    parser.add_argument("--square-mm", type=float, required=True,
                        help="square size measured with a calliper, in mm")
    parser.add_argument("--rotate-portrait", choices=["cw", "ccw"], default=None)
    parser.add_argument("--exclude", nargs="*", default=[], help="file names to leave out")
    args = parser.parse_args()

    os.makedirs(os.path.join(HERE, "detected"), exist_ok=True)
    template = board_points(args.square_mm)
    obj_points, img_points, used, log = [], [], [], []
    target_size = None

    for path in list_images(args.images):
        name = os.path.basename(path)
        tag = f"{os.path.basename(os.path.dirname(path))}/{name}"
        if name in args.exclude:
            log.append(f"SKIP  {tag}: excluded by you")
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # EXIF rotation applied
        if img is None:
            log.append(f"SKIP  {tag}: could not read file")
            continue
        h, w = img.shape[:2]
        portrait = h > w
        if portrait and args.rotate_portrait is None:
            log.append(f"SKIP  {tag}: portrait {w}x{h} (use --rotate-portrait to include)")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray)
        if corners is None:
            log.append(f"FAIL  {tag}: board not detected")
            continue
        if portrait:
            corners = rotate_points(corners, w, h, args.rotate_portrait)
            w, h = h, w
        if target_size is None:
            target_size = (w, h)
        if (w, h) != target_size:
            log.append(f"SKIP  {tag}: size {w}x{h} differs from {target_size[0]}x{target_size[1]}")
            continue
        obj_points.append(template)
        img_points.append(corners)
        used.append(tag)
        preview = cv2.resize(img, None, fx=SEARCH_SCALE, fy=SEARCH_SCALE, interpolation=cv2.INTER_AREA)
        cv2.drawChessboardCorners(preview, PATTERN, corners * SEARCH_SCALE, True)
        cv2.imwrite(os.path.join(HERE, "detected", f"{tag.replace('/', '_')}.jpg"), preview)
        log.append(f"OK    {tag}")

    if len(used) < 10:
        raise SystemExit(f"Only {len(used)} usable photos. Need at least 10, the task asks for 20+.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, target_size, None, None)
    per_image = []
    for i, tag in enumerate(used):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        diff = img_points[i].reshape(-1, 2) - projected.reshape(-1, 2)
        per_image.append((tag, float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))))

    np.savez(os.path.join(HERE, "params.npz"), K=K, dist=dist, image_size=np.array(target_size),
             square_mm=args.square_mm, pattern=np.array(PATTERN))

    lines = [
        f"Photos used:            {len(used)}",
        f"Image size (w x h):     {target_size[0]} x {target_size[1]}",
        f"Square size (mm):       {args.square_mm}",
        f"Reprojection error RMS: {rms:.4f} px",
        "",
        "Camera matrix K:", np.array2string(K, precision=3, suppress_small=True),
        "",
        "Distortion coefficients (k1 k2 p1 p2 k3):",
        np.array2string(dist.ravel(), precision=5, suppress_small=True),
        "",
        "Per-image reprojection error (px), worst first:",
    ]
    lines += [f"  {t:<44s} {e:.4f}" for t, e in sorted(per_image, key=lambda x: -x[1])]
    lines += ["", "Detection log:"] + log
    text = "\n".join(lines)
    with open(os.path.join(HERE, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
