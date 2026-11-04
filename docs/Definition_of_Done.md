# Definition of Done

Written before the object work started. Each box gets a number or a link when it is ticked. A box without evidence counts as unticked.

- [ ] Camera calibrated on 20 or more photos, reprojection error reported with the photo count.
- [ ] Calibration parameters saved to a file that the code loads.
- [ ] Test split list committed before any training (commit link).
- [ ] 70 or more own images of the object, undistorted, labelled with masks (count and labelling tool).
- [ ] Model trained on the train split only. Test split used once, at the end. IoU, mAP@0.5, mAP@0.5:0.95, precision, recall reported.
- [ ] One command takes a new image and prints width and height in mm, with a mask overlay and confidence.
- [ ] "Object not found" and "reference not found" are handled with clear messages.
- [ ] Accuracy table for 10 or more instances, every row shown, plus mean, median and worst case. Raw outputs saved.
- [ ] Raw versus undistorted error shown side by side, with an explanation from my own numbers.
- [ ] MO-1 (accuracy) and MO-2 targets written with how they were measured. Each is met, or missed with the reason.
- [ ] A rule that refuses untrustworthy measurements, with the threshold and the number of refused images.
- [ ] Hardware, camera and image resolution written in the docs. Timings measured 5 times.
- [ ] Seed, library versions and training config pinned. Weights linked with SHA-256 in the README.
- [ ] Plan, Objectives and this file are in `docs/`. README is at the root.
- [ ] Everything I did not finish is listed.
- [ ] Loom recording is 5 minutes or less and answers K1 to K4.
