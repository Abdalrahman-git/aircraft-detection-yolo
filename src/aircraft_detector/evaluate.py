"""Evaluate a checkpoint on a split: ``python -m aircraft_detector.evaluate``.

Also sweeps the confidence threshold, so you can pick an operating point from
data instead of guessing (the original notebook hard-coded 0.65, then 0.50, with
no evidence for either).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import Config, add_config_arguments
from .metrics import collect_detections, evaluate_map, format_metrics, score_detections
from .models import YOLOTiny
from .train import make_loaders
from .utils import get_device, set_seed


def load_checkpoint(path: Path, device: torch.device) -> tuple[YOLOTiny, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    model = YOLOTiny().to(device)
    model.load_state_dict(state.get("model", state))
    model.eval()
    return model, state.get("config", {})


def sweep_confidence(model, loader, device, thresholds, nms_iou) -> list[dict]:
    """Precision/recall/F1 across confidence thresholds.

    Inference runs once and every threshold re-scores the same detections, so
    the sweep costs one forward pass rather than one per threshold.
    """
    per_image = collect_detections(model, loader, device, nms_iou)
    rows = []
    for threshold in thresholds:
        metrics = score_detections(per_image, threshold, ap_iou_thresholds=(0.5,))
        rows.append(
            {
                "conf": round(threshold, 3),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sweep", action="store_true", help="sweep the confidence threshold")
    add_config_arguments(parser, "dataset_dir", "batch_size", "conf_threshold", "seed")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg.add_cli_overrides(args)
    set_seed(cfg.seed)

    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    loader = make_loaders(cfg)[args.split]

    metrics = evaluate_map(
        model, loader, device, cfg.conf_threshold, cfg.nms_iou_threshold
    )
    print(f"{args.split} ({len(loader.dataset)} images): {format_metrics(metrics)}")
    print(
        f"  TP {metrics['true_positives']}  FP {metrics['false_positives']}  "
        f"FN {metrics['false_negatives']}  (of {metrics['num_ground_truth']} objects)"
    )

    if args.sweep:
        rows = sweep_confidence(
            model, loader, device, [i / 20 for i in range(1, 20)], cfg.nms_iou_threshold
        )
        best = max(rows, key=lambda r: r["f1"])
        print("\nconf   precision  recall     f1")
        for row in rows:
            marker = "  <- best F1" if row is best else ""
            print(
                f"{row['conf']:.2f}   {row['precision']:.4f}     "
                f"{row['recall']:.4f}   {row['f1']:.4f}{marker}"
            )
        metrics["sweep"] = rows

    print(json.dumps({k: v for k, v in metrics.items() if k != "sweep"}, indent=2))


if __name__ == "__main__":
    main()
