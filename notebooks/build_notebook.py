"""Generate `aircraft_detection.ipynb`.

The notebook is the narrative; `src/aircraft_detector` is the implementation. It
imports from the package rather than redefining anything, so there is exactly one
version of every function and it is the one the test suite covers.

Generated from this script so the prose stays reviewable as plain text.

    python notebooks/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------
md("""
# Aircraft Detection in Satellite Imagery

Detecting aircraft in very-high-resolution overhead imagery, on the aircraft
class of **NWPU VHR-10**. Two models, compared on equal terms:

1. **A detector written from scratch in PyTorch** — CSP backbone, SPPF neck,
   focal + CIoU loss, NMS and mean average precision, implemented directly with
   no detection framework.
2. **A fine-tuned YOLOv9-S**, from COCO-pretrained weights.

Both are scored by the **same metric code** on the **same seeded split**, so the
comparison is a measurement rather than two numbers from two libraries.

The implementation lives in [`src/aircraft_detector`](../src/aircraft_detector)
and is covered by 95 tests. This notebook imports it rather than restating it —
one version of every function, and it is the tested one.
""")

md("""
---
## 1. Setup
""")

code('''
%matplotlib inline
# Most of this notebook's output is satellite photography, which is far smaller
# as JPEG than as PNG with no visible loss.
%config InlineBackend.figure_format = "jpeg"

import subprocess, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/Abdalrahman-git/aircraft-detection-yolo.git"], check=False)
    %cd aircraft-detection-yolo
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[yolov9]"], check=True)

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["figure.dpi"] = 90
print("project root:", ROOT)
''')

code('''
from aircraft_detector.config import Config
from aircraft_detector.data.dataset import AircraftDataset, build_splits, encode_target, load_boxes
from aircraft_detector.data.prepare import prepare_dataset
from aircraft_detector.models import YOLOTiny
from aircraft_detector.utils import get_device, set_seed

set_seed(42)
DEVICE = get_device()

cfg = Config(dataset_dir=ROOT / "data/aircraft", output_dir=ROOT / "runs/notebook")
print(f"device {DEVICE} | input {cfg.image_size}px | grid {cfg.grid_size}x{cfg.grid_size}")
''')

md("""
---
## 2. The dataset

NWPU VHR-10 is a 10-class geospatial detection dataset: 800 images from Google
Earth and Vaihingen, 650 annotated. Ground truth is one text file per image:

```
(x1,y1),(x2,y2),class_id
```

with `1-airplane, 2-ship, 3-storage tank, 4-baseball diamond, 5-tennis court,
6-basketball court, 7-ground track field, 8-harbor, 9-bridge, 10-vehicle`.

`prepare_dataset` keeps **class 1**, converts the corner boxes to normalised
`(cx, cy, w, h)`, and drops the images with no aircraft in them. It is
research-use only and not redistributed here — set `NWPU_SOURCE`, or let the
Colab branch pull it from Kaggle.
""")

code('''
import os

SOURCE = Path(os.environ.get("NWPU_SOURCE", "nwpu-vhr-10"))
if IN_COLAB and not SOURCE.exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opendatasets"], check=True)
    import opendatasets as od
    od.download("https://www.kaggle.com/datasets/larbisck/nwpu-vhr-10")
    SOURCE = Path("nwpu-vhr-10")

stats = prepare_dataset(SOURCE, cfg.dataset_dir, overwrite=True)
stats
''')

md("""
### What the data actually looks like

Four properties drive every decision that follows, so they are measured rather
than assumed.
""")

code('''
boxes_per_image = [len(load_boxes(p)) for p in sorted((cfg.dataset_dir / "labels").glob("*.txt"))]
all_boxes = np.concatenate([load_boxes(p) for p in sorted((cfg.dataset_dir / "labels").glob("*.txt"))])

print(f"images                {len(boxes_per_image)}")
print(f"aircraft              {len(all_boxes)}")
print(f"per image             mean {np.mean(boxes_per_image):.1f}, max {max(boxes_per_image)}")
print(f"median box (of image) {np.median(all_boxes[:, 2]):.3f} wide x {np.median(all_boxes[:, 3]):.3f} high")
print(f"  -> at {cfg.image_size}px that is "
      f"{np.median(all_boxes[:, 2]) * cfg.image_size:.0f} x {np.median(all_boxes[:, 3]) * cfg.image_size:.0f} px")
print(f"background            {1 - np.mean(boxes_per_image) / cfg.grid_size**2:.1%} of grid cells")
''')

md("""
### Is one box per grid cell enough?

The head predicts a single box per cell, which silently drops a second object
landing in the same cell. With 8.4 aircraft per image that is a real risk, so
`encode_target` reports what it had to drop and we check rather than assume.
""")

code('''
for grid in (10, 20, 40):
    dropped = total = 0
    for label in sorted((cfg.dataset_dir / "labels").glob("*.txt")):
        boxes = load_boxes(label)
        _, lost = encode_target(boxes, grid)
        total += len(boxes)
        dropped += lost
    print(f"grid {grid:>2}x{grid:<2}  keeps {total - dropped}/{total} aircraft ({1 - dropped / total:.1%})")
''')

md("""
At **20x20 nothing is lost**, which is why the input stays at 640 px. At 10x10 it
would be — the same architecture on a smaller input would quietly discard labels.

| | |
|---|---|
| **Tiny objects** | median aircraft ~7% x 11% of the image |
| **Dense scenes** | 8.4 per image, up to 31 |
| **Tiny dataset** | 90 images contain aircraft — the entire training signal |
| **Class imbalance** | 400 cells per image, ~8 positive |
""")

code('''
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, path in zip(axes, sorted((cfg.dataset_dir / "images").glob("*.jpg"))[:3]):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    boxes = load_boxes(cfg.dataset_dir / "labels" / f"{path.stem}.txt")
    ax.imshow(img)
    for cx, cy, bw, bh in boxes:
        ax.add_patch(plt.Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h,
                                   fill=False, edgecolor="#22c55e", lw=1.6))
    ax.set_title(f"{path.name} - {len(boxes)} aircraft", fontsize=10)
    ax.axis("off")
plt.tight_layout(); plt.show()
''')

md("""
---
## 3. Splits and augmentation

Split by **filename** from one seeded shuffle, so the three sets never share an
image and the split is identical on every run. Augmentation is enabled by
building a separate dataset object per split — flipping a shared flag inside
`__getitem__` leaks augmentation into validation as soon as there is more than
one dataloader worker.

Overhead imagery has no canonical "up", so the full dihedral group (flips plus
90-degree rotations) is label-preserving: an 8x expansion of a 62-image training
split for free. Crops are random sub-windows resized back up, so no padded
border is ever introduced, and clipping happens in corner space — clamping a
box's *width* against the border would shrink every box near an edge.
""")

code('''
splits = build_splits(cfg.dataset_dir / "images", cfg.dataset_dir / "labels",
                      cfg.val_fraction, cfg.test_fraction, cfg.seed,
                      cfg.image_size, cfg.grid_size)
print({name: len(ds) for name, ds in splits.items()})
print("augmented:", {name: ds.augment for name, ds in splits.items()})
assert not set(splits["train"].files) & set(splits["test"].files)
''')

code('''
set_seed(0)
aug = AircraftDataset(cfg.dataset_dir / "images", cfg.dataset_dir / "labels",
                      splits["train"].files, cfg.image_size, cfg.grid_size, augment=True)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax in axes:
    image, _, boxes = aug[0]
    ax.imshow(image.permute(1, 2, 0).numpy())
    for cx, cy, bw, bh in boxes.tolist():
        s = cfg.image_size
        ax.add_patch(plt.Rectangle(((cx - bw / 2) * s, (cy - bh / 2) * s), bw * s, bh * s,
                                   fill=False, edgecolor="#22c55e", lw=1.5))
    ax.axis("off")
fig.suptitle("One training image, four augmentations", fontsize=11)
plt.tight_layout(); plt.show()
set_seed(cfg.seed)
''')

md("""
---
## 4. The model

```
input 640x640x3
  -> stem     6x6 conv, stride 2, padding 2     -> 320x320x16
  -> CSP x4   downsample, split, concat, fuse   ->  20x20x256
  -> SPPF     three chained 5x5 max-pools       ->  20x20x256
  -> head     1x1 conv -> 5 channels, sigmoid   ->  20x20x5
```

Each cell predicts `(tx, ty, w, h, objectness)` — offsets inside the cell, and a
size normalised to the whole image.

**The stem padding is not arbitrary.** A 6x6 stride-2 convolution padded by
`kernel // 2 = 3` emits `in/2 + 1` rows, not `in/2`: 640 px would produce a
**21x21** grid while the decoder assumes the grid tiles the image exactly,
putting every predicted box off by a fraction of a cell. Even kernels need
`(kernel - stride) // 2`.

Objectness bias starts at −4.6 (sigmoid ~0.01) so the model begins by predicting
nothing anywhere, rather than spending its first epochs unlearning a uniform
positive prior.
""")

code('''
model = YOLOTiny()
with torch.no_grad():
    out = model(torch.zeros(1, 3, cfg.image_size, cfg.image_size))

print(f"parameters    {sum(p.numel() for p in model.parameters()):,}")
print(f"output        {tuple(out.shape)}")
print(f"objectness at init  mean {out[..., 4].mean():.4f}  (biased towards empty)")
assert out.shape[1] == cfg.grid_size
''')

md("""
### Why CIoU rather than IoU

Plain IoU is exactly zero for any pair of non-overlapping boxes, so it produces
**no gradient** in precisely the situation that dominates early training. CIoU
adds centre-distance and aspect-ratio terms that still rank a near miss above a
far one:
""")

code('''
from aircraft_detector.boxes import box_iou, complete_iou, cxcywh_to_xyxy

target = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
for label, pred in [("near miss", torch.tensor([[0.7, 0.5, 0.1, 0.1]])),
                    ("far miss",  torch.tensor([[0.95, 0.5, 0.1, 0.1]]))]:
    iou = float(box_iou(cxcywh_to_xyxy(pred), cxcywh_to_xyxy(target)))
    print(f"{label:<10} IoU {iou:.3f}   CIoU {float(complete_iou(pred, target)):+.3f}")
print("\\nIoU cannot tell these apart; CIoU can, so the box loss keeps a gradient.")
''')

md("""
---
## 5. Training

Linear warmup into cosine decay as a **single** schedule. Attaching a `LambdaLR`
warmup and a `CosineAnnealingLR` to one optimizer and stepping whichever matches
the epoch does not work — `LambdaLR` rescales `base_lr` while
`CosineAnnealingLR` tracks its own step count, so the two fight and the curve is
neither intended shape.

Checkpoint selection uses validation **AP@0.5** rather than F1, because AP is
independent of the confidence threshold and so cannot be flattered by a lucky
calibration.
""")

code('''
from aircraft_detector.train import train

summary = train(cfg)
summary
''')

code('''
import json as _json
from aircraft_detector.report import plot_history

history = _json.loads((cfg.output_dir / "history.json").read_text())
plot_history(history, cfg.output_dir / "curves.png")
from IPython.display import Image as IPyImage
IPyImage(str(cfg.output_dir / "curves.png"))
''')

md("""
### The confidence threshold is not a detail

Look at the gap between AP@0.5 and F1 above. High AP with low F1 means the model
**ranks** boxes well but its objectness is calibrated badly — the default
threshold is far too low for it. So sweep validation for the F1-maximising
value rather than guessing.
""")

code('''
from aircraft_detector.benchmark import THRESHOLD_GRID, tune_threshold
from aircraft_detector.evaluate import load_checkpoint
from aircraft_detector.metrics import collect_detections, format_metrics, score_detections
from aircraft_detector.train import make_loaders

model, _ = load_checkpoint(cfg.output_dir / "best.pt", DEVICE)
loaders = make_loaders(cfg)

val_dets = collect_detections(model, loaders["val"], DEVICE, cfg.nms_iou_threshold)
test_dets = collect_detections(model, loaders["test"], DEVICE, cfg.nms_iou_threshold)
BEST_CONF = tune_threshold(val_dets)

sweep = [(t, score_detections(val_dets, t, ap_iou_thresholds=(0.5,))) for t in THRESHOLD_GRID]
plt.figure(figsize=(7, 4))
for key, colour in [("precision", "#2563eb"), ("recall", "#f59e0b"), ("f1", "#10b981")]:
    plt.plot([t for t, _ in sweep], [m[key] for _, m in sweep], color=colour, label=key)
plt.axvline(BEST_CONF, ls="--", c="grey", lw=1)
plt.xlabel("confidence threshold"); plt.ylim(0, 1); plt.legend(frameon=False); plt.grid(alpha=.3)
plt.title(f"Operating point chosen on validation: {BEST_CONF}"); plt.show()

print("test @ default 0.35:", format_metrics(score_detections(test_dets, 0.35)))
print(f"test @ tuned {BEST_CONF}   :", format_metrics(score_detections(test_dets, BEST_CONF)))
''')

md("""
Same weights, same images — only the threshold moved. That is why the benchmark
below picks each model's operating point on validation instead of forcing both
to share one, which would compare calibration rather than detection quality.
""")

md("""
---
## 6. YOLOv9

YOLOv9 contributes **GELAN** (a generalised ELAN/CSP backbone) and **PGI** (an
auxiliary reversible branch used during training and discarded at inference).

**It is fine-tuned, not trained from scratch, and that is deliberate.** There are
62 training images. YOLOv9-S is ~7M parameters against this baseline's 1.1M, and
PGI pays off on deep networks with real data volume. From random initialisation
on 62 images it would lose to the smaller model — which would be a fact about
the dataset, not about YOLOv9.

### The default learning rate destroys the model

Ultralytics' `optimizer="auto"` selects AdamW at `lr0=0.002`. On this dataset
that does not underfit or overfit — it **erases the pretrained weights**:

| epoch | 2 | 3 | 4 | 5 | 6 | 14 |
|---|---|---|---|---|---|---|
| val mAP@0.5 | **0.937** | 0.023 | 0.833 | 0.090 | 0.000 | 0.002 |

The peak at epoch 2 is the model still inside its 3-epoch warmup. The moment the
full rate applies, COCO features are overwritten by gradients from 62 images and
never recover. `build_train_kwargs` pins AdamW at `lr0=3e-4` with cosine decay
for that reason.
""")

code('''
from aircraft_detector.yolov9.export import export_for_ultralytics
from aircraft_detector.yolov9.train import build_train_kwargs
from aircraft_detector.yolov9.adapter import load_ultralytics_model

# Export OUR seeded splits. Left to itself Ultralytics invents its own split, and
# a model evaluated on different images is not comparable to anything.
export = export_for_ultralytics(cfg, ROOT / "data/aircraft_yolo", overwrite=True)
print(export["counts"])

yolo = load_ultralytics_model("yolov9s.pt")
kwargs = build_train_kwargs(
    data_yaml=ROOT / "data/aircraft_yolo/data.yaml",
    epochs=80, imgsz=cfg.image_size, batch=8, seed=cfg.seed,
    device=0 if torch.cuda.is_available() else "cpu", workers=0,
    project=ROOT / "runs/yolov9", name="notebook", patience=20,
)
yolo.train(**kwargs)
YOLO_WEIGHTS = Path(yolo.trainer.save_dir) / "weights" / "best.pt"
print("best weights:", YOLO_WEIGHTS)
''')

md("""
---
## 7. Head to head

Ultralytics' `mAP50` and this project's `ap50` are not the same quantity — they
differ in interpolation, NMS defaults and matching. So neither framework's
summary is used. `benchmark` reduces both models to `(boxes, scores,
ground_truth)` and scores them with the same `score_detections`, on the same
images, each at the threshold it earned on validation.

Both are measured in normalised original-image coordinates: a pure resize leaves
normalised boxes unchanged, and Ultralytics reports in original pixels, so each
model keeps its native preprocessing while the units stay identical.
""")

code('''
from aircraft_detector.benchmark import benchmark, format_table

report = benchmark(cfg, cfg.output_dir / "best.pt", YOLO_WEIGHTS, split="test",
                   imgsz=cfg.image_size, figure=ROOT / "runs/notebook/comparison.jpg")

print(f"test split: {report['images']} images, {report['objects']} aircraft\\n")
for name, entry in report["results"].items():
    print(f"  {name}: conf >= {entry['conf_threshold']:.2f} ({entry['threshold_source']})")
print()
print(format_table(report["results"]))
''')

code('''
IPyImage(str(ROOT / "runs/notebook/comparison.jpg"))
''')

md("""
---
## 8. What this shows

**Both models find the aircraft; they differ on false positives.** Recall is
close. The from-scratch model fires on hangars, terminal buildings and storage
tanks, and that is what costs it precision.

**The wider gap is AP@0.5:0.95, which rewards tight localisation.** A
single-scale 20x20 grid regressing one box per cell is structurally outmatched
by a multi-scale anchor-based head. Finding the aircraft is the easy half;
putting the box exactly on it is the half that needs the architecture.

**Most of the remaining difference is the pretrained backbone, not the
architecture.** On 62 training images, COCO initialisation is worth more than
any design choice available on either side.

**Caveat:** the test split is 14 images. These numbers are real on this split,
but it is a small sample and none of them should be read as a general claim.

### Things that only showed up by running it

- A 6x6 stride-2 convolution padded by `k // 2` emits `in/2 + 1` rows, quietly
  misaligning the entire grid against the decoder.
- Clamping a box's *width* against the frame border shrinks every box near an
  edge, because a full width is not a half-width.
- A framework default (`lr0=0.002`) that is sensible at COCO scale can destroy a
  model at 62 images, and only per-epoch validation makes that visible.
- An untuned confidence threshold moved this project's reported F1 from 0.27 to
  0.70 without touching the weights.
""")


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.split("\n")
        payload = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
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
    destination = Path(__file__).parent / "aircraft_detection.ipynb"
    destination.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {destination} ({len(CELLS)} cells, {n_code} code)")
