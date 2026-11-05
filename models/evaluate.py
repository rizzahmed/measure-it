"""Evaluate the trained model on the test split, once, at the very end.

Usage (from the repository root):
    python models/evaluate.py

The test split must not be used to pick checkpoints or tune anything, so this script
refuses to run a second time. Use --split val to look at validation numbers instead.
"""
import argparse
import json
import os

import torch

from common import BookCardDataset, build_model, evaluate_split, load_config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--config", default=os.path.join(HERE, "config.json"))
    parser.add_argument("--run", default=os.path.join(HERE, "runs", "default"))
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args()

    out_path = os.path.join(args.run, f"{args.split}_metrics.json")
    if args.split == "test" and os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists. The test split is evaluated once; "
                         "delete the file only if you are starting a whole new run.")
    cfg = load_config(args.config)
    device = torch.device("cpu")
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(args.run, "best.pt"), map_location=device))
    model.to(device)

    dataset = BookCardDataset(args.root, args.split, cfg["image_scale"], augment=False)
    metrics = evaluate_split(model, dataset, device, cfg["score_threshold"], cfg["iou_threshold"])
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
