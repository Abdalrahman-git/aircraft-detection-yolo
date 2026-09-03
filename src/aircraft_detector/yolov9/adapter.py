"""Score an Ultralytics model with *our* metric code, on *our* splits.

Comparing this project's AP against the number Ultralytics prints would be
meaningless: the two use different interpolation, different NMS defaults and a
different matching rule. So instead of trusting either framework's summary, both
models are reduced to the same intermediate form -

    (boxes_xyxy, scores, ground_truth_xyxy)   # all normalised to [0, 1]

- and fed through :func:`aircraft_detector.metrics.score_detections`.

**The shared coordinate space.** The baseline resizes each whole image to a
square, and normalised ``(cx, cy, w, h)`` is invariant under a pure resize - no
crop, no padding, so the fractional position of a box does not move. Ultralytics
letterboxes instead, but reports predictions back in *original* pixel
coordinates, which divide by the original width and height into the same
normalised frame. Each model therefore keeps its own native preprocessing while
both are scored in identical units.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..boxes import cxcywh_to_xyxy
from ..data.dataset import load_boxes
from ..metrics import Detections


def load_ultralytics_model(weights: str | Path):
    """Import lazily so the core package never requires `ultralytics`."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The YOLOv9 track needs the optional extra: pip install -e \".[yolov9]\""
        ) from exc
    return YOLO(str(weights))


def collect_ultralytics_detections(
    model,
    image_paths: list[Path],
    label_dir: Path,
    conf_threshold: float = 1e-3,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: str | None = None,
) -> Detections:
    """Run an Ultralytics model over images and return normalised detections.

    ``conf_threshold`` defaults to 1e-3 so the full precision/recall curve is
    available to the scorer; filtering to an operating point happens later.
    """
    label_dir = Path(label_dir)
    paths = [Path(p) for p in image_paths]

    results = model.predict(
        source=[str(p) for p in paths],
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
        device=device,
    )
    if len(results) != len(paths):
        raise RuntimeError(
            f"Ultralytics returned {len(results)} results for {len(paths)} images"
        )

    per_image: Detections = []
    for path, result in zip(paths, results, strict=True):
        raw = result.boxes
        if raw is None or len(raw) == 0:
            boxes = torch.zeros((0, 4))
            scores = torch.zeros((0,))
        else:
            # `xyxy` is already rescaled to the original image, and `orig_shape`
            # is (height, width) - so no second read of the file is needed.
            height, width = result.orig_shape
            scale = torch.tensor([width, height, width, height], dtype=torch.float32)
            boxes = (raw.xyxy.detach().cpu().float() / scale).clamp(0, 1)
            scores = raw.conf.detach().cpu().float()

        gt = load_boxes(label_dir / f"{path.stem}.txt")
        gt_xyxy = (
            cxcywh_to_xyxy(torch.from_numpy(gt)).clamp(0, 1) if len(gt) else torch.zeros((0, 4))
        )
        per_image.append((boxes, scores, gt_xyxy))

    return per_image
