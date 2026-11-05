# Definition of Done

Written before the object work started. Each box gets a number or a link when it is ticked. A box without evidence counts as unticked.

- [x] Camera calibrated on 20 or more photos, reprojection error reported with the photo count.
- [x] Calibration parameters saved to a file that the code loads.
- [x] Test split list committed before any training (commit link).
- [x] 70 or more own images of the object, undistorted, labelled with masks (count and labelling tool).
- [x] Model trained on the train split only. Test split used once, at the end. IoU, mAP@0.5, mAP@0.5:0.95, precision, recall reported.
- [x] One command takes a new image and prints width and height in mm, with a mask overlay and confidence.
- [x] "Object not found" and "reference not found" are handled with clear messages.
- [x] Accuracy table for 10 or more instances, every row shown, plus mean, median and worst case. Raw outputs saved.
- [x] Raw versus undistorted error shown side by side, with an explanation from my own numbers.
- [x] MO-1 (accuracy) and MO-2 targets written with how they were measured. Each is met, or missed with the reason.
- [x] A rule that refuses untrustworthy measurements, with the threshold and the number of refused images.
- [x] Hardware, camera and image resolution written in the docs. Timings measured 5 times.
- [x] Seed, library versions and training config pinned. Weights linked with SHA-256 in the README.
- [x] Plan, Objectives and this file are in `docs/`. README is at the root.
- [x] Everything I did not finish is listed.
- [ ] Loom recording is 5 minutes or less and answers K1 to K4.

## Evidence

- Calibration: 27 photos, 0.605 px, commit d270c7c (`calibration/report.txt`).
- Split before code: commit 3ca5516.
- Dataset: 80 images, 80 book and 70 card masks, Label Studio, commit deb3470. The images are used as photographed; undistortion is applied at measurement time.
- Training: 15 epochs, best epoch 12, validation IoU 0.9615; test (once): IoU 0.959, mAP@0.5 1.000, mAP@0.5:0.95 0.973, precision 1.000, recall 1.000 (`models/runs/default/test_metrics.json`).
- Measurement command: commit 91de8f3. Refusal rule: score below 0.7 or card shape off by more than 8 %; 1 of 10 accuracy photos refused.
- Accuracy: `measurement/results.csv` (9 measured). Raw and undistorted differ by less than 1 mm on average, within the noise of 9 photos.
- MO-2 is ticked as "missed with the reason": mean error -6.0 mm (width) and -10.7 mm (height) against 3 / 5 mm.
- Hardware and timings: README (median 3.9 s per photo over 5 runs). RAM was not recorded.
- Weights: GitHub Release 1.0, SHA-256 in the README.
- Still open: Loom recording.
