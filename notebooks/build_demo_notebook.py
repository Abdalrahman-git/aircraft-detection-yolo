"""Generate `aircraft_detection_demo.ipynb`.

The notebook is generated from this script so it stays in sync with the package
and is committed without execution output, keeping the repository small and the
diffs readable.

    python notebooks/build_demo_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Aircraft Detection on NWPU VHR-10

A walkthrough of the pipeline in `src/aircraft_detector`: prepare the data, train
a from-scratch YOLO-style detector, evaluate it, and look at the predictions.

All the logic lives in the package, so this notebook stays short and every cell
here matches what the CLI does. Run it top to bottom on Colab or locally.""",
    ),
    (
        "markdown",
        """## 1. Setup

On Colab, clone the repo and install the dependencies. Locally, just
`pip install -e .` from the project root.""",
    ),
    (
        "code",
        """import sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    subprocess.run(
        ["git", "clone", "https://github.com/Abdalrahman-git/aircraft-detection-yolo.git"],
        check=False,
    )
    %cd aircraft-detection-yolo
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
print("project root:", ROOT)""",
    ),
    (
        "markdown",
        """## 2. Get the dataset

NWPU VHR-10 is a 10-class remote sensing detection dataset (800 images from
Google Earth and Vaihingen). It is research-use only, so it is not vendored in
this repo -- download it from Kaggle, or point `--source` at your own copy.""",
    ),
    (
        "code",
        """# On Colab, pull the dataset from Kaggle (needs a kaggle.json API token).
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opendatasets"], check=True)
    import opendatasets as od

    od.download("https://www.kaggle.com/datasets/larbisck/nwpu-vhr-10")
    SOURCE = Path("nwpu-vhr-10")
else:
    SOURCE = Path("path/to/NWPU VHR-10 dataset")  # <- edit this

print("source:", SOURCE)""",
    ),
    (
        "markdown",
        """## 3. Prepare the aircraft subset

NWPU VHR-10 labels ten classes as `(x1,y1),(x2,y2),class_id`. We keep **class 1,
airplane**, convert the corner boxes to normalised YOLO `cx cy w h`, and drop
every image with no aircraft in it.""",
    ),
    (
        "code",
        """from aircraft_detector.data.prepare import prepare_dataset

stats = prepare_dataset(SOURCE, ROOT / "data/aircraft", class_id=1, overwrite=True)
stats""",
    ),
    (
        "markdown",
        """Expect **90 images / 757 aircraft** -- an average of 8.4 per image, and a small
dataset. That size drives most of the design choices below: heavy augmentation,
a ~1.1M parameter model, and a held-out test split that never sees augmentation.""",
    ),
    (
        "markdown",
        """## 4. Look at the data

Ground-truth boxes over a few training images.""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt
from aircraft_detector.data import build_splits

splits = build_splits(
    ROOT / "data/aircraft/images", ROOT / "data/aircraft/labels",
    val_fraction=0.15, test_fraction=0.15, seed=42, image_size=640, grid_size=20,
)
print({k: len(v) for k, v in splits.items()})

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, idx in zip(axes, range(3)):
    image, _, boxes = splits["val"][idx]
    ax.imshow(image.permute(1, 2, 0).numpy())
    for cx, cy, bw, bh in boxes.tolist():
        ax.add_patch(plt.Rectangle(
            ((cx - bw / 2) * 640, (cy - bh / 2) * 640), bw * 640, bh * 640,
            fill=False, edgecolor="#22c55e", linewidth=1.5))
    ax.set_title(f"{len(boxes)} aircraft"); ax.axis("off")
plt.tight_layout()""",
    ),
    (
        "markdown",
        """## 5. The model

`stem -> 4x CSP -> SPPF -> 1x1 head`, total stride 32, so a 640x640 image becomes
a 20x20 grid. Each cell predicts one box: `(tx, ty, w, h, objectness)`.

At 20x20 every one of the 757 aircraft in this dataset lands in its own cell, so
the one-box-per-cell head loses nothing here -- worth checking before trusting
this design on another class.""",
    ),
    (
        "code",
        """import torch
from aircraft_detector.models import YOLOTiny

model = YOLOTiny()
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print("output shape:", tuple(model(torch.zeros(1, 3, 640, 640)).shape))""",
    ),
    (
        "markdown",
        """## 6. Train

Focal loss on background objectness (99.7% of cells are empty) plus CIoU box
regression. Validation runs every 10 epochs and the best AP@0.5 checkpoint is
kept.

Drop `epochs` if you just want to see it move; the committed results use 200.""",
    ),
    (
        "code",
        """from aircraft_detector.config import Config
from aircraft_detector.train import train

cfg = Config(
    dataset_dir=ROOT / "data/aircraft",
    output_dir=ROOT / "runs/train",
    epochs=200,
    batch_size=4 if not torch.cuda.is_available() else 8,
    num_workers=0,
)
summary = train(cfg)
summary""",
    ),
    (
        "markdown",
        """## 7. Evaluate

`evaluate_map` scores predictions against the **raw** ground-truth boxes rather
than the grid-encoded target, so the encoding cannot hide a missed object.
Average precision sweeps the whole confidence range; precision/recall/F1 are
reported at the operating threshold.""",
    ),
    (
        "code",
        """from aircraft_detector.evaluate import load_checkpoint, sweep_confidence
from aircraft_detector.metrics import evaluate_map, format_metrics
from aircraft_detector.train import make_loaders
from aircraft_detector.utils import get_device

device = get_device()
model, _ = load_checkpoint(ROOT / "runs/train/best.pt", device)
loaders = make_loaders(cfg)

metrics = evaluate_map(model, loaders["test"], device, cfg.conf_threshold, cfg.nms_iou_threshold)
print("test:", format_metrics(metrics))
metrics""",
    ),
    (
        "markdown",
        """### Choosing the confidence threshold

Sweep it instead of guessing -- the value that maximises F1 on validation is the
one to deploy.""",
    ),
    (
        "code",
        """rows = sweep_confidence(model, loaders["val"], device, [i / 20 for i in range(1, 20)], 0.45)
best = max(rows, key=lambda r: r["f1"])
print("best F1 at conf =", best["conf"], "->", best)

plt.figure(figsize=(7, 4))
for key, colour in [("precision", "#2563eb"), ("recall", "#f59e0b"), ("f1", "#10b981")]:
    plt.plot([r["conf"] for r in rows], [r[key] for r in rows], label=key, color=colour)
plt.axvline(best["conf"], ls="--", c="grey", lw=1)
plt.xlabel("confidence threshold"); plt.ylim(0, 1); plt.legend(frameon=False); plt.grid(alpha=.3)""",
    ),
    (
        "markdown",
        """## 8. Predictions

Green is ground truth, red is the model.""",
    ),
    (
        "code",
        """from aircraft_detector.report import plot_samples

plot_samples(model, loaders["test"].dataset, device, ROOT / "docs/assets/detections.jpg", cfg)
from IPython.display import Image as IPyImage
IPyImage(str(ROOT / "docs/assets/detections.jpg"))""",
    ),
    (
        "markdown",
        """## 9. Fine-tune YOLOv9 and benchmark against it

The baseline above tells us what a detector does. This section tells us how good
it actually is, by putting a modern production detector next to it.

**Why fine-tune rather than train YOLOv9 from scratch:** there are 62 training
images. YOLOv9-S is 7.3M parameters against the baseline's 1.1M, and PGI (its
train-time auxiliary branch) pays off on deep networks with real data volume.
From random weights on 62 images it would lose to the smaller baseline -- that
would be a fact about the dataset, not about YOLOv9. From COCO-pretrained
weights the backbone already knows generic edges and shapes.

This is the cell that wants a GPU (Runtime -> Change runtime type -> T4).""",
    ),
    (
        "code",
        """subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

from aircraft_detector.yolov9.export import export_for_ultralytics

# Export OUR seeded splits into the Ultralytics layout. Without this step
# Ultralytics invents its own split and the comparison becomes meaningless.
summary = export_for_ultralytics(cfg, ROOT / "data/aircraft_yolo", overwrite=True)
print(summary["counts"])""",
    ),
    (
        "code",
        """from ultralytics import YOLO
from aircraft_detector.yolov9.train import build_train_kwargs

yolo = YOLO("yolov9s.pt")
kwargs = build_train_kwargs(
    data_yaml=ROOT / "data/aircraft_yolo/data.yaml",
    epochs=100, imgsz=640, batch=8, seed=cfg.seed,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=2, project=ROOT / "runs/yolov9", name="finetune", patience=30,
)
yolo.train(**kwargs)""",
    ),
    (
        "markdown",
        """### Head to head

Both models scored by the **same** `score_detections`, on the **same** test
split, at the **same** operating point. Ultralytics' own `mAP50` is deliberately
not used -- it differs from this project's `ap50` in interpolation, NMS defaults
and matching, so quoting the two side by side would be comparing definitions
rather than models.""",
    ),
    (
        "code",
        """from aircraft_detector.benchmark import benchmark, format_table

report = benchmark(
    cfg,
    baseline_checkpoint=ROOT / "runs/train/best.pt",
    yolov9_weights=ROOT / "runs/yolov9/finetune/weights/best.pt",
    split="test",
)
print(f"{report['images']} images, {report['objects']} aircraft, conf >= {report['conf_threshold']}\\n")
print(format_table(report["results"]))""",
    ),
    (
        "markdown",
        """## 10. Run on a new image

The same thing from the command line:

```bash
python -m aircraft_detector.predict \\
    --checkpoint runs/train/best.pt \\
    --source path/to/image.jpg \\
    --output-dir runs/predict
```""",
    ),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.split("\n")
        payload = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": payload}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    destination = Path(__file__).parent / "aircraft_detection_demo.ipynb"
    destination.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
