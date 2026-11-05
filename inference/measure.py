"""Measure a flat book (width, height in mm) from one photo with a bank card lying beside it.

Usage (from the repository root):
    python inference/measure.py photo.jpg [--out inference/output] [--raw]

Steps: read photo (EXIF applied) -> undistort -> Mask R-CNN gives book and card masks
-> minAreaRect of each mask -> mm per pixel from the card -> book size in mm.
Prints a JSON line per photo; writes an overlay image to --out.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.join(ROOT, "calibration"))
from common import build_model, load_config  # noqa: E402
from undistort import load_params, undistort_image  # noqa: E402

CARD_LONG_MM, CARD_SHORT_MM = 85.6, 54.0  # ISO/IEC 7810 ID-1
MIN_SCORE = 0.7            # below this the detection is not trusted
MAX_CARD_ASPECT_ERR = 0.08 # card seen as a rectangle must be within 8% of 85.6/54.0


def load_model(run_dir, cfg):
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    return model.eval()


@torch.no_grad()
def detect(model, bgr, cfg):
    """Return {label: (score, full-resolution binary mask)} for the best book and card."""
    h, w = bgr.shape[:2]
    scale = cfg["image_scale"]
    small = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).permute(2, 0, 1)
    out = model([tensor])[0]
    found = {}
    for label, name in ((1, "book"), (2, "card")):
        idx = [i for i in range(len(out["labels"])) if int(out["labels"][i]) == label]
        if not idx:
            continue
        best = max(idx, key=lambda i: float(out["scores"][i]))
        prob = out["masks"][best, 0].numpy()
        # upsample the probability map (not the binary mask) so edges are not 4-pixel steps
        mask = (cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
        found[name] = (float(out["scores"][best]), mask)
    return found


def rect_of(mask):
    """Rotated rectangle around the largest blob of the mask: (centre, (w, h), angle)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.minAreaRect(max(contours, key=cv2.contourArea))


def measure(found):
    """Convert pixels to mm. Needs card and book in the same plane (card lying on / next to the book)."""
    for name in ("book", "card"):
        if name not in found:
            return {"status": "refused", "reason": f"{name} not found"}
        if found[name][0] < MIN_SCORE:
            return {"status": "refused", "reason": f"{name} score {found[name][0]:.2f} < {MIN_SCORE}"}
    card = rect_of(found["card"][1])
    book = rect_of(found["book"][1])
    card_long, card_short = sorted(card[1], reverse=True)
    aspect_err = abs(card_long / card_short / (CARD_LONG_MM / CARD_SHORT_MM) - 1)
    if aspect_err > MAX_CARD_ASPECT_ERR:
        return {"status": "refused", "reason": f"card shape off by {aspect_err:.0%} (partly hidden or camera tilted)"}
    # THE pixel -> mm line: the card's two sides in pixels correspond to 85.6 + 54.0 mm.
    mm_per_px = (CARD_LONG_MM + CARD_SHORT_MM) / (card_long + card_short)
    book_long, book_short = sorted(book[1], reverse=True)
    return {"status": "ok", "width_mm": round(book_short * mm_per_px, 1),
            "height_mm": round(book_long * mm_per_px, 1), "mm_per_px": round(mm_per_px, 5),
            "book_score": round(found["book"][0], 3), "card_score": round(found["card"][0], 3),
            "card_aspect_error": round(aspect_err, 3)}, book, card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--out", default=os.path.join(HERE, "output"))
    parser.add_argument("--raw", action="store_true", help="skip undistortion (for comparison only)")
    parser.add_argument("--run", default=os.path.join(ROOT, "models", "runs", "default"))
    args = parser.parse_args()

    cfg = load_config(os.path.join(ROOT, "models", "config.json"))
    K, dist, size = load_params()
    model = load_model(args.run, cfg)
    os.makedirs(args.out, exist_ok=True)
    for path in args.images:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # EXIF rotation applied
        if bgr is None:
            print(json.dumps({"image": path, "status": "refused", "reason": "cannot read file"}))
            continue
        if (bgr.shape[1], bgr.shape[0]) != size:
            print(json.dumps({"image": path, "status": "refused",
                              "reason": f"size {bgr.shape[1]}x{bgr.shape[0]} differs from calibration {size[0]}x{size[1]}"}))
            continue
        if not args.raw:
            bgr = undistort_image(bgr, K, dist)
        found = detect(model, bgr, cfg)
        result = measure(found)
        if isinstance(result, dict):
            print(json.dumps({"image": os.path.basename(path), **result}))
            continue
        info, book, card = result
        print(json.dumps({"image": os.path.basename(path), **info}))
        for rect, colour in ((book, (0, 200, 0)), (card, (0, 140, 255))):
            cv2.polylines(bgr, [cv2.boxPoints(rect).astype(np.int32)], True, colour, 6)
        text = f"{info['width_mm']} x {info['height_mm']} mm"
        cv2.putText(bgr, text, (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
        cv2.imwrite(os.path.join(args.out, os.path.basename(path)), bgr)


if __name__ == "__main__":
    main()
