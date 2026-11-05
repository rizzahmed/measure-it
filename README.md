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

Object: book. Images: to be filled in. Split: 56 / 16 / 8 (see `dataset/splits/`). Labelling tool: to be filled in.

## How to run each stage

Calibration (already run, 27 photos, 0.61 px reprojection error):

    python calibration/calibrate.py --images calibration/images --square-mm <measured> --rotate-portrait cw

The remaining stages are added here as they are built.

## Hardware

To be filled in (CPU, GPU, RAM, operating system).
