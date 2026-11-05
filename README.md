# Measure It

Photograph a book lying flat next to a bank card and report the book's real width and height in millimetres.

## Setup decisions

| Item | Choice |
|---|---|
| Object | A closed book, lying flat, cover facing the camera |
| Reference object | A bank card lying on the book cover, same plane |
| Camera | Xiaomi 23021RAAEG rear camera, 4080 x 2296, landscape, focus locked |
| Calibration board | calib.io checkerboard, 9 x 11 squares (10 x 8 inner corners), A4 print |
| Ground truth tool | Ruler / measuring scale, 1 mm graduation (reading precision about 0.5 mm) |

## Measured sizes (filled in by me, by hand)

| Thing | Width (mm) | Height (mm) | How measured |
|---|---|---|---|
| Bank card, my ruler reading | 87 | 55 | Ruler, 1 mm graduation, read from the photo and by eye. Reading error about 1 mm |
| Bank card, value used for scale | 85.6 | 54.0 | Standard ID-1 card size (ISO/IEC 7810: 85.60 x 53.98 mm). More precise than my ruler, and my reading is within my reading error of it |
| Book | 137 | 210 | Ruler, read once, to be re-checked |

Width and height definition for the book: width is the short side of the closed cover (137 mm), height is the long side (210 mm).

## Dataset summary

Object: one book (the same copy in every photo) with one bank card. 80 own photos, 4080 x 2296, labelled by me in Label Studio
(run locally) with polygon masks: 80 book masks and 70 card masks (10 photos have no card). Split 56 / 16 / 8 for train / validation / test, fixed
before any training (`dataset/splits/`, seed 42). The photos are not in the repository because the card number is readable in them
(available on request). Masks are in `dataset/annotations/instances_full.json`.

## Results

| What | Result |
|---|---|
| Camera calibration | 27 photos, reprojection error 0.605 px (`calibration/report.txt`) |
| Segmentation, test split (8 images, 15 masks, evaluated once) | mean mask IoU 0.959, mAP@0.5 1.000, mAP@0.5:0.95 0.973, precision 1.000, recall 1.000 (`models/runs/default/test_metrics.json`) |
| Millimetres, 10 new photos, card on the cover, undistorted | 9 measured, 1 refused. Mean error: width -6.0 mm, height -10.7 mm. Worst case 9.7 mm and 16.6 mm. Target (3 mm / 5 mm) **missed** |
| Same, without undistortion | width -5.2 mm, height -10.0 mm (no better or worse than undistorted at this sample size) |
| Card lying next to the book, not on it | width +7.7 mm, height +9.0 mm, heights between 211 and 231 mm. Breaking condition, see below |

Every row of the tables is in `measurement/results.csv` (card on the cover) and `measurement/results_card_beside.csv`.
Reference values: book 137 x 210 mm by ruler, card 85.6 x 54.0 mm (standard size).

## How it works

`inference/measure.py`: read the photo (EXIF rotation applied), undistort with `calibration/params.npz`, run Mask R-CNN at 0.25 scale,
take the best book mask and the best card mask, fit a rotated rectangle around each, and compute
`mm_per_px = (85.6 + 54.0) / (card_long_px + card_short_px)`. The book's short side times `mm_per_px` is the width, the long side is the height.
The model only finds masks. The millimetres come from this one line, which is only valid when the card and the book cover are in the same plane
and the camera looks roughly straight down.

It answers "refused" with a reason when the book or the card is missing, when a score is below 0.7, or when the card's shape differs
from 85.6 : 54.0 by more than 8 % (partly covered or strongly tilted). In the 10 accuracy photos, 1 was refused (card score 0.63).

## What breaks it

- **Card not in the plane of the cover.** With the card on the table next to a thick book, the cover is closer to the camera and looks bigger:
  +5 % error on all photos. Put the card on the cover.
- **Tilted camera or book.** Height changed by up to 20 mm between photos of the same book.
- **Soft or uneven surfaces** (bed, cloth): the book sinks and tilts. Several training photos were taken like that.

## What I could not explain

With the card on the cover, every photo is 4-5 % too small, in width and in height, and the hand-drawn masks of my 70 labelled photos
give the same size (134.8 x 203.3 mm), so the model is not the main cause. Candidates I did not have time to separate: my ruler reading of the
book (read once), the real size of my card (I read 87 x 55 mm, I used the standard 85.6 x 54.0), small differences in calibration between photo sets.

## Reproducing

    pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

    python calibration/calibrate.py --images calibration/images --square-mm 19.5 --rotate-portrait cw   # calibration
    python models/train.py                                                                               # about 3 h on CPU
    python models/evaluate.py                                                                            # test split, once
    python inference/measure.py photo.jpg                                                                # one photo -> mm
    python measurement/accuracy.py measurement/photos --width 137 --height 210                           # accuracy table
    python inference/timing.py photo.jpg                                                                 # 5 timing runs

Seed 42, config in `models/config.json`, library versions in `requirements.txt` (Python 3.11.6).

## Weights

`models/runs/default/best.pt` (176 MB) is not in git. It is a GitHub Release asset:
[release 1.0](https://github.com/rizzahmed/measure-it/releases/tag/1.0), direct download `https://github.com/rizzahmed/measure-it/releases/download/1.0/best.pt`. Put it in `models/runs/default/` before running `evaluate.py` or `measure.py`; check it with `sha256sum` (or `certutil -hashfile best.pt SHA256` on Windows).
SHA-256: `71dd28cb980f4d1701977f304f18fb426bf2ed85b567363095a79fb75550b9e8`

## Hardware

Laptop, Windows 10 (build 26300), Intel 11th-gen i-series CPU, 8 threads, no GPU (CPU only). RAM: not recorded.
Camera: Xiaomi 23021RAAEG, 4080 x 2296. Training: 15 epochs, about 3.1 h. Time per photo, whole pipeline on CPU: median 3.9 s over 5 runs (1.8 s first run, 3.7-4.2 s after; `inference/timing.py`).

## Where AI helped and what I did myself

An AI assistant (Claude) helped with the plan, with the code for calibration, training, evaluation and measurement, and with explanations.
I took all photos, did the calibration shooting, labelled all 80 images, ran the calibration and the training on my laptop, measured the
book and card by ruler, and ran every result in this document.

## Not finished

- Calibration: the checkerboard square size (19.5 mm) is an estimate, and 2 calibration photos were left out by mistake.
- The 4-5 % bias is not explained and the target MO-2 was missed.
- Ground truth was read once with a ruler; no repeated readings.
- Validation set is only 16 images, test set only 8, all from one book.
