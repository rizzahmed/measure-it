"""Dataset, model and metrics shared by train.py and evaluate.py."""
import json
import os

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

CLASS_NAMES = {1: "book", 2: "card"}


def load_config(path):
    with open(path) as fh:
        return json.load(fh)


class BookCardDataset(torch.utils.data.Dataset):
    """Images from dataset/images with the polygons in instances_full.json.

    Images and polygons are scaled by `scale` so Mask R-CNN can train on a CPU.
    """

    def __init__(self, root, split, scale, augment):
        with open(os.path.join(root, "dataset", "annotations", "instances_full.json")) as fh:
            coco = json.load(fh)
        with open(os.path.join(root, "dataset", "splits", f"{split}.txt")) as fh:
            wanted = set(fh.read().split())
        self.root, self.scale, self.augment = root, scale, augment
        self.images = [i for i in coco["images"] if i["file_name"] in wanted]
        if len(self.images) != len(wanted):
            raise SystemExit(f"{split}: {len(wanted)} names in the split list, "
                             f"{len(self.images)} found in the annotations")
        self.anns = {}
        for ann in coco["annotations"]:
            self.anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def load(self, index):
        """Return the scaled RGB image and the instances of one image as numpy arrays."""
        info = self.images[index]
        path = os.path.join(self.root, "dataset", "images", info["file_name"])
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"Cannot read {path}")
        h, w = bgr.shape[:2]
        size = (round(w * self.scale), round(h * self.scale))
        rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        masks, labels = [], []
        for ann in self.anns.get(info["id"], []):
            poly = np.array(ann["segmentation"][0]).reshape(-1, 2) * self.scale
            m = np.zeros(rgb.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
            masks.append(m)
            labels.append(ann["category_id"])
        return rgb, np.stack(masks), np.array(labels, np.int64)

    def __getitem__(self, index):
        rgb, masks, labels = self.load(index)
        if self.augment:
            if np.random.rand() < 0.5:
                rgb, masks = rgb[:, ::-1], masks[:, :, ::-1]
            if np.random.rand() < 0.5:
                rgb, masks = rgb[::-1], masks[:, ::-1]
            gain, bias = np.random.uniform(0.8, 1.2), np.random.uniform(-20, 20)
            rgb = np.clip(rgb.astype(np.float32) * gain + bias, 0, 255)
        image = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1)
        masks = torch.from_numpy(np.ascontiguousarray(masks))
        boxes = []
        for m in masks.numpy():
            ys, xs = np.where(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.from_numpy(labels), "masks": masks,
                  "image_id": self.images[index]["id"]}
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(cfg, pretrained):
    """Mask R-CNN with a ResNet-50 FPN backbone and new heads for book and card."""
    width, height = cfg["full_image_size"]
    scale = cfg["image_scale"]
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None,
                                  min_size=round(height * scale), max_size=round(width * scale))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg["num_classes"])
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, cfg["num_classes"])
    return model


@torch.no_grad()
def predict(model, dataset, device):
    """Run the model on every image of the dataset. Returns per image: masks, labels, scores."""
    model.eval()
    results = []
    for i in range(len(dataset)):
        image, _ = dataset[i]
        out = model([image.to(device)])[0]
        masks = (out["masks"][:, 0] > 0.5).cpu().numpy().astype(np.uint8)
        results.append({"masks": masks, "labels": out["labels"].cpu().numpy(),
                        "scores": out["scores"].cpu().numpy()})
    return results


def _rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def evaluate_split(model, dataset, device, score_threshold, iou_threshold):
    """Mask IoU, mAP@0.5, mAP@0.5:0.95, precision and recall on one dataset."""
    predictions = predict(model, dataset, device)
    gt = {"images": [], "annotations": [], "categories": [{"id": k, "name": v} for k, v in CLASS_NAMES.items()]}
    dets, gt_by_image = [], []
    ann_id = 1
    for i, pred in enumerate(predictions):
        _, masks, labels = dataset.load(i)
        gt["images"].append({"id": i, "height": masks.shape[1], "width": masks.shape[2]})
        gt_by_image.append((masks, labels))
        for m, lab in zip(masks, labels):
            gt["annotations"].append({"id": ann_id, "image_id": i, "category_id": int(lab),
                                      "segmentation": _rle(m), "area": float(m.sum()), "iscrowd": 0})
            ann_id += 1
        for m, lab, score in zip(pred["masks"], pred["labels"], pred["scores"]):
            dets.append({"image_id": i, "category_id": int(lab), "segmentation": _rle(m), "score": float(score)})

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    if dets:
        evaluator = COCOeval(coco_gt, coco_gt.loadRes(dets), "segm")
        evaluator.evaluate(), evaluator.accumulate(), evaluator.summarize()
        map_5095, map_50 = float(evaluator.stats[0]), float(evaluator.stats[1])
    else:
        map_5095 = map_50 = 0.0

    tp = fp = fn = 0
    best_iou = []
    for (masks, labels), pred in zip(gt_by_image, predictions):
        keep = pred["scores"] >= score_threshold
        p_masks, p_labels = pred["masks"][keep], pred["labels"][keep]
        order = np.argsort(-pred["scores"][keep])
        matched = set()
        for j in order:
            same = [g for g in range(len(labels)) if labels[g] == p_labels[j] and g not in matched]
            ious = [float(mask_utils.iou([_rle(p_masks[j])], [_rle(masks[g])], [0])[0][0]) for g in same]
            if ious and max(ious) >= iou_threshold:
                matched.add(same[int(np.argmax(ious))])
                tp += 1
            else:
                fp += 1
        fn += len(labels) - len(matched)
        for g in range(len(labels)):
            ious = [float(mask_utils.iou([_rle(p_masks[j])], [_rle(masks[g])], [0])[0][0])
                    for j in range(len(p_labels)) if p_labels[j] == labels[g]]
            best_iou.append(max(ious) if ious else 0.0)
    return {"images": len(dataset), "ground_truth_instances": int(sum(len(l) for _, l in gt_by_image)),
            "mean_mask_iou": float(np.mean(best_iou)), "map_50": map_50, "map_50_95": map_5095,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn,
            "score_threshold": score_threshold, "iou_threshold": iou_threshold}
