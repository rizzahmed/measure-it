"""Fine-tune Mask R-CNN on the book and card masks (train split only).

Usage (from the repository root):
    python models/train.py

The validation split picks the best epoch. The test split is never opened here.
Output goes to models/runs/default/: best.pt, best.pt.sha256, train_log.csv, run_info.json.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time

import numpy as np
import torch
import torchvision

from common import BookCardDataset, build_model, collate, evaluate_split, load_config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--config", default=os.path.join(HERE, "config.json"))
    parser.add_argument("--out", default=os.path.join(HERE, "runs", "default"))
    parser.add_argument("--no-pretrained", action="store_true", help="skip the COCO weights (offline smoke tests)")
    parser.add_argument("--max-iters", type=int, default=0, help="stop each epoch early (smoke tests only)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cpu")

    train_set = BookCardDataset(args.root, "train", cfg["image_scale"], augment=True)
    val_set = BookCardDataset(args.root, "val", cfg["image_scale"], augment=False)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(cfg["seed"]))

    model = build_model(cfg, pretrained=cfg["pretrained"] and not args.no_pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["learning_rate"], momentum=cfg["momentum"],
                                weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg["lr_milestones"], cfg["lr_gamma"])

    log_path = os.path.join(args.out, "train_log.csv")
    best_path = os.path.join(args.out, "best.pt")
    best_map, best_epoch = -1.0, 0
    with open(log_path, "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "train_loss", "val_map_50", "val_map_50_95", "val_mean_mask_iou", "seconds"])
        for epoch in range(1, cfg["epochs"] + 1):
            started = time.time()
            model.train()
            losses = []
            for step, (images, targets) in enumerate(loader):
                images = [i.to(device) for i in images]
                targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
                loss = sum(model(images, targets).values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if args.max_iters and step + 1 >= args.max_iters:
                    break
            scheduler.step()
            metrics = evaluate_split(model, val_set, device, cfg["score_threshold"], cfg["iou_threshold"])
            seconds = time.time() - started
            log.writerow([epoch, f"{np.mean(losses):.4f}", f"{metrics['map_50']:.4f}",
                          f"{metrics['map_50_95']:.4f}", f"{metrics['mean_mask_iou']:.4f}", f"{seconds:.0f}"])
            fh.flush()
            print(f"epoch {epoch}/{cfg['epochs']}  loss {np.mean(losses):.3f}  "
                  f"val mAP50 {metrics['map_50']:.3f}  mAP50-95 {metrics['map_50_95']:.3f}  "
                  f"IoU {metrics['mean_mask_iou']:.3f}  {seconds:.0f}s", flush=True)
            if metrics["map_50_95"] > best_map:
                best_map, best_epoch = metrics["map_50_95"], epoch
                torch.save(model.state_dict(), best_path)

    digest = sha256(best_path)
    with open(best_path + ".sha256", "w") as fh:
        fh.write(f"{digest}  best.pt\n")
    with open(os.path.join(args.out, "run_info.json"), "w") as fh:
        json.dump({"config": cfg, "best_epoch": best_epoch, "best_val_map_50_95": best_map,
                   "weights_sha256": digest, "torch": torch.__version__,
                   "torchvision": torchvision.__version__, "python": platform.python_version(),
                   "machine": platform.platform(), "processor": platform.processor()}, fh, indent=2)
    print(f"best epoch {best_epoch}, val mAP50-95 {best_map:.3f}\nSHA-256 {digest}")


if __name__ == "__main__":
    main()
