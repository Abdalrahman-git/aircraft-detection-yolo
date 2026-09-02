"""Fine-tune a pretrained YOLOv9 on the aircraft split.

    python -m aircraft_detector.yolov9.train --epochs 100

Why fine-tune rather than train from scratch: this dataset has 62 training
images. YOLOv9-S is roughly 7M parameters (YOLOv9-C is 25M) against the baseline's
1.1M, and its headline contribution - PGI, an auxiliary reversible branch used
only during training - pays off on deep networks with substantial data. Started
from random weights on 62 images it would underperform the small baseline. From
COCO-pretrained weights the backbone already encodes generic shape and edge
structure, which is exactly what a 62-image dataset cannot teach it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import Config, add_config_arguments
from .adapter import load_ultralytics_model
from .export import export_for_ultralytics


def build_train_kwargs(
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    seed: int,
    device: str | None,
    workers: int,
    project: Path,
    name: str,
    patience: int,
    lr0: float = 3e-4,
) -> dict:
    return {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": seed,
        "deterministic": True,
        "device": device,
        "workers": workers,
        # Absolute, deliberately. Ultralytics resolves a *relative* `project`
        # against its own configured `runs_dir`, so "runs/yolov9" silently lands
        # in "runs/detect/runs/yolov9" and the weights are not where you expect.
        "project": str(Path(project).resolve()),
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "patience": patience,
        "val": True,
        "plots": True,
        # --- learning rate: the single setting that decides whether this works ---
        # Ultralytics' `optimizer="auto"` picks AdamW at lr0=0.002. That is tuned
        # for datasets orders of magnitude larger than this one. Measured here on
        # 62 images (8 iterations/epoch), it reached mAP50 0.937 during warmup and
        # then destroyed the pretrained weights the moment the full rate kicked in:
        #
        #   epoch    2      3      4     5     6     14
        #   mAP50  0.937  0.023  0.833  0.09  0.000  0.002
        #
        # That is catastrophic forgetting, not underfitting - more epochs never
        # recover it. Fine-tuning a pretrained backbone on a tiny dataset needs a
        # rate small enough to adapt the features rather than overwrite them.
        "optimizer": "AdamW",
        "lr0": lr0,
        "lrf": 0.01,
        "cos_lr": True,
        "warmup_epochs": 3.0,
        # Overhead imagery has no canonical "up", so a vertical flip is as valid
        # as a horizontal one. Ultralytics leaves flipud at 0 because that is
        # wrong for natural photos; here it is free extra data.
        "fliplr": 0.5,
        "flipud": 0.5,
        # Left at 0 deliberately. The baseline augments with exact 90-degree
        # rotations, which Ultralytics cannot express; its `degrees` option
        # rotates by an arbitrary angle, and re-fitting an axis-aligned box
        # around a rotated one inflates every label. Flips only, rather than
        # quietly corrupting the boxes.
        "degrees": 0.0,
        # Mosaic is the strongest augmentation available for a tiny dataset, but
        # it distorts scale statistics, so it is disabled for the final epochs.
        # Scaled to the run length: a fixed 10 would disable mosaic for the
        # entire run on any schedule shorter than that.
        "mosaic": 1.0,
        "close_mosaic": min(10, epochs // 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--weights", default="yolov9s.pt", help="pretrained checkpoint to start from"
    )
    parser.add_argument("--yolo-dataset", type=Path, default=Path("data/aircraft_yolo"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--lr0", type=float, default=3e-4, help="initial LR (see the note in build_train_kwargs)"
    )
    parser.add_argument("--project", type=Path, default=Path("runs/yolov9"))
    parser.add_argument("--name", default="finetune")
    parser.add_argument(
        "--skip-export", action="store_true", help="reuse an existing exported dataset"
    )
    add_config_arguments(parser, "dataset_dir", "seed")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg.add_cli_overrides(args)

    if not args.skip_export:
        summary = export_for_ultralytics(cfg, args.yolo_dataset, overwrite=True)
        print(f"Exported splits: {summary['counts']}")
    data_yaml = Path(args.yolo_dataset).resolve() / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found - run without --skip-export first")

    model = load_ultralytics_model(args.weights)
    kwargs = build_train_kwargs(
        data_yaml,
        args.epochs,
        args.imgsz,
        args.batch,
        cfg.seed,
        args.device,
        args.workers,
        args.project,
        args.name,
        args.patience,
        args.lr0,
    )
    print(json.dumps({k: v for k, v in kwargs.items() if k != "data"}, indent=2))
    model.train(**kwargs)

    # Ask the trainer where it actually saved, rather than reconstructing the
    # path and hoping it matches.
    save_dir = Path(getattr(model.trainer, "save_dir", Path(kwargs["project"]) / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    if not best.exists():
        print(f"WARNING: expected weights at {best}, but the file is missing")
    print(
        "Compare against the from-scratch baseline with:\n"
        f"  python -m aircraft_detector.benchmark --yolov9-weights {best}"
    )


if __name__ == "__main__":
    main()
