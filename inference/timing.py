"""Time the full pipeline (read, undistort, model, measurement) on one photo, 5 runs.

Usage: python inference/timing.py photo.jpg
"""
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as m  # noqa: E402

path = sys.argv[1]
cfg = m.load_config(os.path.join(m.ROOT, "models", "config.json"))
K, dist, size = m.load_params()
model = m.load_model(os.path.join(m.ROOT, "models", "runs", "default"), cfg)
times = []
for _ in range(5):
    start = time.time()
    img = m.undistort_image(cv2.imread(path), K, dist)
    m.measure(m.detect(model, img, cfg))
    times.append(time.time() - start)
print("seconds per image:", [round(t, 1) for t in times], "median", round(statistics.median(times), 1))
