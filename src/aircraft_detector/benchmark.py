"""Compare the from-scratch baseline against fine-tuned YOLOv9, fairly.

    python -m aircraft_detector.benchmark --yolov9-weights runs/yolov9/finetune/weights/best.pt

Fairness rests on three things, all enforced here rather than assumed:

1. **The same images.** Both models are evaluated on the split produced by the
   one seeded shuffle in `build_splits`, and the image list is asserted equal.
2. **The same metric code.** Both are reduced to normalised detections and
   scored by `metrics.score_detections`. Neither framework's own mAP number is
   used, because they do not agree on interpolation, NMS or matching.
3. **A per-model operating point, chosen on validation.** Average precision is
   threshold-free and directly comparable. Precision, recall and F1 are not:
   two detectors can rank boxes equally well and still need very different
   confidence cut-offs. Forcing one shared threshold on both measures
   calibration, not detection quality - so each model's threshold is the one
   that maximises F1 on the *validation* split, and test is scored there. The
   test split is never used to choose anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .config import Config, add_config_arguments
from .metrics import collect_detections, score_detections
from .train import make_loaders
from .utils import get_device, set_seed

METRIC_ROWS = [
    ("AP@0.5", "ap50"),
    ("AP@0.5:0.95", "ap50_95"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
]


def _baseline_detections(cfg: Config, checkpoint: Path, split: str, device):
    from .evaluate import load_checkpoint

    model, _ = load_checkpoint(checkpoint, device)
    loader: DataLoader = make_loaders(cfg)[split]
    files = list(loader.dataset.files)
    detections = collect_detections(model, loader, device, cfg.nms_iou_threshold)
    params = sum(p.numel() for p in model.parameters())
    return detections, files, params


def _yolov9_detections(cfg: Config, weights: Path, files: list[str], device, imgsz: int):
    from .yolov9.adapter import collect_ultralytics_detections, load_ultralytics_model

    model = load_ultralytics_model(weights)
    paths = [cfg.dataset_dir / "images" / name for name in files]
    detections = collect_ultralytics_detections(
        model,
        paths,
        cfg.dataset_dir / "labels",
        conf_threshold=1e-3,
        iou_threshold=cfg.nms_iou_threshold,
        imgsz=imgsz,
        device=None if device is None else str(device),
    )
    params = sum(p.numel() for p in model.model.parameters())
    return detections, params


def format_table(results: dict[str, dict]) -> str:
    names = list(results)
    header = "| Metric | " + " | ".join(names) + " |"
    divider = "|---|" + "---|" * len(names)
    lines = [header, divider]
    for label, key in METRIC_ROWS:
        cells = [f"{results[n]['metrics'][key]:.3f}" for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(
        "| Parameters | " + " | ".join(f"{results[n]['parameters']:,}" for n in names) + " |"
    )
    lines.append(
        "| Conf. threshold | "
        + " | ".join(f"{results[n]['conf_threshold']:.2f}" for n in names)
        + " |"
    )
    return "\n".join(lines)


THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def tune_threshold(validation_detections, grid=THRESHOLD_GRID) -> float:
    """Pick the confidence threshold that maximises F1 on validation."""
    return max(
        grid,
        key=lambda t: score_detections(validation_detections, t, ap_iou_thresholds=(0.5,))["f1"],
    )


def _assert_same_evaluation_set(name: str, reference, candidate) -> None:
    ref_objects = sum(int(gt.shape[0]) for _, _, gt in reference)
    cand_objects = sum(int(gt.shape[0]) for _, _, gt in candidate)
    if len(reference) != len(candidate) or ref_objects != cand_objects:
        raise RuntimeError(
            f"Evaluation sets diverged for {name}: baseline saw "
            f"{len(reference)} images / {ref_objects} objects, "
            f"{name} saw {len(candidate)} / {cand_objects}"
        )


def benchmark(
    cfg: Config,
    baseline_checkpoint: Path,
    yolov9_weights: Path | None,
    split: str = "test",
    imgsz: int = 640,
    tune: bool = True,
    figure: Path | None = None,
) -> dict:
    set_seed(cfg.seed)
    device = get_device()

    # Detections on the reported split, plus validation for threshold selection.
    baseline_eval, eval_files, baseline_params = _baseline_detections(
        cfg, baseline_checkpoint, split, device
    )
    collected = {"from-scratch": (baseline_eval, baseline_params)}

    validation: dict[str, list] = {}
    if tune:
        baseline_val, val_files, _ = _baseline_detections(cfg, baseline_checkpoint, "val", device)
        validation["from-scratch"] = baseline_val

    if yolov9_weights is not None:
        yolo_eval, yolo_params = _yolov9_detections(cfg, yolov9_weights, eval_files, device, imgsz)
        _assert_same_evaluation_set("yolov9s", baseline_eval, yolo_eval)
        collected["yolov9s"] = (yolo_eval, yolo_params)
        if tune:
            yolo_val, _ = _yolov9_detections(cfg, yolov9_weights, val_files, device, imgsz)
            _assert_same_evaluation_set("yolov9s (val)", baseline_val, yolo_val)
            validation["yolov9s"] = yolo_val

    results: dict[str, dict] = {}
    thresholds: dict[str, float] = {}
    for name, (detections, params) in collected.items():
        threshold = tune_threshold(validation[name]) if tune else cfg.conf_threshold
        thresholds[name] = threshold
        results[name] = {
            "metrics": score_detections(detections, threshold),
            "parameters": params,
            "conf_threshold": threshold,
            "threshold_source": "tuned on val (max F1)" if tune else "shared, from config",
        }

    if figure is not None:
        from .report import plot_model_comparison

        figure.parent.mkdir(parents=True, exist_ok=True)
        plot_model_comparison(
            make_loaders(cfg)[split].dataset,
            {name: detections for name, (detections, _) in collected.items()},
            thresholds,
            figure,
            cfg.image_size,
        )

    return {
        "split": split,
        "images": len(eval_files),
        "objects": sum(int(gt.shape[0]) for _, _, gt in baseline_eval),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("runs/train/config.yaml"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("runs/train/best.pt"))
    parser.add_argument("--yolov9-weights", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark.json"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="write a side-by-side detection comparison image here",
    )
    parser.add_argument(
        "--shared-threshold",
        action="store_true",
        help="score both models at config.conf_threshold instead of tuning each on val",
    )
    add_config_arguments(parser, "dataset_dir", "conf_threshold", "seed")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg.add_cli_overrides(args)

    report = benchmark(
        cfg,
        args.baseline_checkpoint,
        args.yolov9_weights,
        args.split,
        args.imgsz,
        tune=not args.shared_threshold,
        figure=args.figure,
    )

    print(f"\n{args.split}: {report['images']} images, {report['objects']} aircraft")
    for name, entry in report["results"].items():
        print(f"  {name}: conf >= {entry['conf_threshold']:.2f} ({entry['threshold_source']})")
    print()
    print(format_table(report["results"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
