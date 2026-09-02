"""Generate the figures used in the README from a finished training run.

    python -m aircraft_detector.report --run runs/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .config import Config
from .metrics import collect_detections, evaluate_map
from .postprocess import decode_predictions
from .utils import get_device

PALETTE = {"loss": "#2563eb", "ap": "#f59e0b", "f1": "#10b981", "grid": "#e5e7eb"}


def plot_history(history: list[dict], dest: Path) -> None:
    """Training loss alongside validation AP50 / F1."""
    epochs = [r["epoch"] for r in history]
    losses = [r["train_total"] for r in history]
    eval_epochs = [r["epoch"] for r in history if "val_ap50" in r]
    ap50 = [r["val_ap50"] for r in history if "val_ap50" in r]
    f1 = [r["val_f1"] for r in history if "val_f1" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, losses, color=PALETTE["loss"], linewidth=1.8)
    ax1.set_title("Training loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("total loss")

    ax2.plot(eval_epochs, ap50, color=PALETTE["ap"], marker="o", ms=3, label="val AP@0.5")
    ax2.plot(eval_epochs, f1, color=PALETTE["f1"], marker="s", ms=3, label="val F1")
    ax2.set_title("Validation metrics")
    ax2.set_xlabel("epoch")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False)

    for ax in (ax1, ax2):
        ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def plot_samples(model, dataset, device, dest: Path, cfg: Config, n: int = 6) -> None:
    """Grid of test images with predictions (red) over ground truth (green)."""
    n = min(n, len(dataset))
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        if idx >= n:
            continue
        image, _, gt = dataset[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))[0].detach().cpu()
        boxes, scores = decode_predictions(pred, cfg.conf_threshold, cfg.nms_iou_threshold)

        ax.imshow(image.permute(1, 2, 0).numpy())
        size = cfg.image_size
        for cx, cy, bw, bh in gt.tolist():
            ax.add_patch(
                plt.Rectangle(
                    ((cx - bw / 2) * size, (cy - bh / 2) * size),
                    bw * size,
                    bh * size,
                    fill=False,
                    edgecolor="#22c55e",
                    linewidth=1.6,
                )
            )
        for (x1, y1, x2, y2), score in zip(boxes.tolist(), scores.tolist(), strict=True):
            ax.add_patch(
                plt.Rectangle(
                    (x1 * size, y1 * size),
                    (x2 - x1) * size,
                    (y2 - y1) * size,
                    fill=False,
                    edgecolor="#ef4444",
                    linewidth=1.6,
                )
            )
            ax.text(
                x1 * size,
                max(y1 * size - 4, 8),
                f"{score:.2f}",
                color="white",
                fontsize=7,
                bbox={"facecolor": "#ef4444", "edgecolor": "none", "pad": 1},
            )
        ax.set_title(f"{len(boxes)} predicted / {len(gt)} actual", fontsize=9)

    fig.suptitle("green = ground truth,  red = prediction", fontsize=10, y=0.995)
    fig.tight_layout()
    # JPEG, not PNG: these panels are photographs, and PNG makes them ~10x larger
    # than a repository should carry for a README image.
    fig.savefig(dest, dpi=110, pil_kwargs={"quality": 82})
    plt.close(fig)


def _draw(ax, image, gt, boxes, scores, size: int, title: str) -> None:
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.axis("off")
    for cx, cy, bw, bh in gt.tolist():
        ax.add_patch(
            plt.Rectangle(
                ((cx - bw / 2) * size, (cy - bh / 2) * size),
                bw * size,
                bh * size,
                fill=False,
                edgecolor="#22c55e",
                linewidth=1.5,
            )
        )
    for x1, y1, x2, y2 in boxes.tolist():
        ax.add_patch(
            plt.Rectangle(
                (x1 * size, y1 * size),
                (x2 - x1) * size,
                (y2 - y1) * size,
                fill=False,
                edgecolor="#ef4444",
                linewidth=1.5,
            )
        )
    ax.set_title(title, fontsize=9)


def plot_model_comparison(
    dataset,
    detections_by_model: dict[str, list],
    thresholds: dict[str, float],
    dest: Path,
    image_size: int,
    n: int = 3,
) -> None:
    """One row per image, one column per model, at each model's own threshold.

    `detections_by_model` holds the `(boxes, scores, gt)` triples the benchmark
    already collected, in dataset order - so this renders exactly the detections
    that produced the reported numbers, rather than re-running the models with
    possibly different settings.
    """
    names = list(detections_by_model)
    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, len(names), figsize=(4.6 * len(names), 4.6 * n), squeeze=False)

    for row in range(n):
        image, _, gt = dataset[row]
        for col, name in enumerate(names):
            boxes, scores, _ = detections_by_model[name][row]
            keep = scores >= thresholds[name]
            kept_boxes, kept_scores = boxes[keep], scores[keep]
            _draw(
                axes[row][col],
                image,
                gt,
                kept_boxes,
                kept_scores,
                image_size,
                f"{name}: {len(kept_boxes)} predicted / {len(gt)} actual",
            )

    fig.suptitle("green = ground truth,  red = prediction", fontsize=10, y=0.997)
    fig.tight_layout()
    fig.savefig(dest, dpi=110, pil_kwargs={"quality": 82})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/train"))
    parser.add_argument("--assets-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="override the operating point (default: tuned on the validation split)",
    )
    args = parser.parse_args()

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_yaml(args.run / "config.yaml")

    history = json.loads((args.run / "history.json").read_text(encoding="utf-8"))
    plot_history(history, args.assets_dir / "training_curves.png")
    print(f"wrote {args.assets_dir / 'training_curves.png'}")

    checkpoint = args.run / "best.pt"
    if not checkpoint.exists():
        return

    from .evaluate import load_checkpoint
    from .train import make_loaders

    device = get_device()
    model, _ = load_checkpoint(checkpoint, device)
    loaders = make_loaders(cfg)

    # Draw and score at the model's real operating point. The config default is
    # only a starting guess, and a badly calibrated model at the wrong threshold
    # produces a picture of hundreds of false positives that says nothing useful.
    if args.conf_threshold is not None:
        cfg.conf_threshold = args.conf_threshold
    else:
        from .benchmark import tune_threshold

        val_detections = collect_detections(
            model, loaders["val"], device, cfg.nms_iou_threshold
        )
        cfg.conf_threshold = tune_threshold(val_detections)
    print(f"operating point: conf >= {cfg.conf_threshold:.2f}")

    plot_samples(model, loaders["test"].dataset, device, args.assets_dir / "detections.jpg", cfg)
    print(f"wrote {args.assets_dir / 'detections.jpg'}")

    results = {
        split: evaluate_map(
            model, loaders[split], device, cfg.conf_threshold, cfg.nms_iou_threshold
        )
        for split in ("val", "test")
    }
    (args.run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
