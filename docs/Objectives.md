# Objectives

Written before the accuracy test. Targets are fixed here and are not changed after the results.

## Model objective (MO-1): segmentation
On the held-out test split (8 images, 15 instances), mean mask IoU >= 0.90 and mask mAP@0.5 >= 0.95
for the classes book and card.

Result (test split, evaluated once): mean mask IoU 0.959, mAP@0.5 1.000, mAP@0.5:0.95 0.973,
precision 1.000, recall 1.000 (15 TP, 0 FP, 0 FN, score 0.5, IoU 0.5). File: `models/runs/default/test_metrics.json`.
Caveat: 8 images is a small test set; the images come from the same book, card and similar tables as the training data.

## Measurement objective (MO-2): millimetres
On at least 10 new photos of the same book on a hard flat table, none used for training, the mean absolute
error against the ruler is <= 3 mm for width and <= 5 mm for height. Photos where the system refuses to
answer are counted and reported, not removed.

Result: **missed.** 9 of 10 new photos were measured (1 refused), card lying on the book cover, undistorted:
mean error width -6.0 mm (MAE 6.0 mm, 4.4 %), height -10.7 mm (MAE 10.7 mm, 5.1 %), worst case 9.7 mm and 16.6 mm.
The measured sizes are about 4-5 % smaller than the ruler values on every photo (same sign each time), so this is a
steady bias and not random noise. The cause is not settled (see the README, section "What I could not explain").
Raw table: `measurement/results.csv`. The targets above were not changed after seeing the numbers.

## Known limits (stated before the test)
- Ruler ground truth has about 1 mm reading precision; the book values 137 x 210 mm were read once and are re-checked.
- One mask pixel at the training scale (0.25) is about 4 full-resolution pixels, about 0.4 mm, so a mask edge error of one pixel is below 1 mm.
- Scale comes from the card (ISO 85.6 x 54.0 mm); it is only valid if card and book lie in the same plane and the camera looks roughly straight down.
- On 70 labelled images the hand-drawn masks alone give 134.8 x 203.3 mm (spread 3.6 mm), so an error of a few mm may come from the setup and not from the model.
