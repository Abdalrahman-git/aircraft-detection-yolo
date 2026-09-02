"""Run the detector on images and save annotated copies.

    python -m aircraft_detector.predict --checkpoint runs/train/best.pt --source img.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .config import Config
from .postprocess import decode_predictions
from .utils import get_device

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def gather_images(source: Path) -> list[Path]:
    source = Path(source)
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(source)


def preprocess(path: Path, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    original = Image.open(path).convert("RGB")
    resized = original.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0), original


def draw_detections(
    image: Image.Image, boxes: torch.Tensor, scores: torch.Tensor, width: int = 3
) -> Image.Image:
    """Draw normalised xyxy boxes onto a copy of the image at its native size."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for (x1, y1, x2, y2), score in zip(boxes.tolist(), scores.tolist(), strict=True):
        box = [x1 * w, y1 * h, x2 * w, y2 * h]
        draw.rectangle(box, outline=(255, 60, 60), width=width)
        label = f"{score:.2f}"
        tx, ty = box[0], max(box[1] - 12, 0)
        draw.rectangle([tx, ty, tx + 6 * len(label) + 4, ty + 12], fill=(255, 60, 60))
        draw.text((tx + 2, ty), label, fill=(255, 255, 255))
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="image file or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predict"))
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    args = parser.parse_args()

    from .evaluate import load_checkpoint  # local import keeps the CLI startup light

    device = get_device()
    model, saved_config = load_checkpoint(args.checkpoint, device)

    cfg = Config()
    for key in ("image_size", "conf_threshold", "nms_iou_threshold"):
        if key in saved_config:
            setattr(cfg, key, saved_config[key])
    if args.conf_threshold is not None:
        cfg.conf_threshold = args.conf_threshold
    if args.nms_iou_threshold is not None:
        cfg.nms_iou_threshold = args.nms_iou_threshold

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = gather_images(args.source)
    if not paths:
        raise SystemExit(f"No images found under {args.source}")

    total = 0
    for path in paths:
        tensor, original = preprocess(path, cfg.image_size)
        pred = model(tensor.to(device))[0].detach().cpu()
        boxes, scores = decode_predictions(pred, cfg.conf_threshold, cfg.nms_iou_threshold)
        annotated = draw_detections(original, boxes, scores)
        destination = args.output_dir / f"{path.stem}_pred.jpg"
        annotated.save(destination, quality=95)
        total += len(boxes)
        print(f"{path.name}: {len(boxes)} aircraft -> {destination}")

    print(f"\n{total} detections across {len(paths)} image(s) at conf>={cfg.conf_threshold}")


if __name__ == "__main__":
    main()
