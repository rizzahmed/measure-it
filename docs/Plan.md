# Plan: Measure It

Branch: `dev-test01`. Written before any dataset or model code was committed.

## Goal

Photograph one flat object and report its real width and height in millimetres, with a mask overlay and a confidence value. Prove the accuracy against a calliper or tape.

## Choices made so far

| Topic | Choice | Reason |
|---|---|---|
| Camera | Xiaomi 23021RAAEG, rear camera, 4080 x 2296, landscape | One camera for everything. Focus and zoom locked for object photos. |
| Calibration board | calib.io checkerboard, 9 x 11 squares (10 x 8 inner corners), printed on A4 | Above the task minimum of 7 x 9 inner corners. |
| Object | One flat, rectangular, matte object (named in README) | Flat top face means the object and the reference share one plane. |
| Reference object | Rigid flat card of known size, measured by me with my own tool | Gives the pixel to millimetre scale. Printed size is not trusted. |
| Ground truth | Calliper if available, otherwise tape (stated in README) | The tool and its precision are written next to every accuracy number. |
| Model | Mask R-CNN (torchvision), pretrained backbone, fine-tuned on my images | Not YOLO. It gives instance masks, so mAP and IoU come out naturally. |
| Dataset | 80 images, named `obj_001.jpg` to `obj_080.jpg`, all taken by me | Task needs 70 or more. The extra 10 cover images that get dropped. |
| Split | 56 train / 16 val / 8 test, fixed in `dataset/splits/` | About 70 / 20 / 10. Made with `random.Random(42)` over the file names, committed before any photo or code. |

## Order and time budget (8 hours of work)

| Part | Hours | Items |
|---|---|---|
| Plan, split list, setup | 0.5 | |
| Camera calibration | 1.0 | 1 |
| Photograph the object and undistort | 1.0 | 2 |
| Label masks | 1.5 | 2 |
| Train the model (runs in the background) | 1.0 | 3 |
| Image to millimetres command | 1.0 | 4 |
| Accuracy against calliper, raw versus undistorted | 1.0 | 5, 6 |
| One target, refusal rule | 0.5 | 7, 9 |
| Documents, README, weights hash, pull request | 0.5 | 10 |

Items 1 to 4 come first. Item 8 only starts once item 5 is finished.

## What is already done, said honestly

Camera calibration was done before this Plan was written, with an AI assistant helping, on 27 checkerboard photos (0.61 px reprojection error). The calibration scripts are in the working folder but are not committed yet. They will be committed after this Plan and the split list.

## Decided to skip

- Item 8 (a condition that breaks the system) if time runs out. It will then be listed as not reached.
- Any model comparison. One model, one training config.

## Decision record

Filled in at the end, once the work shows what was taken, what was rejected and what rejecting it cost.
