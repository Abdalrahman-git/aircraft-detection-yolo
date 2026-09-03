"""Convert the NWPU VHR-10 annotations into single-class YOLO labels for airplanes.

NWPU VHR-10 ground-truth lines look like::

    (563,478),(630,573),1

where ``(x1,y1)`` is the top-left corner, ``(x2,y2)`` the bottom-right corner and
the trailing integer is the class id. Per the dataset readme the ids are
1-airplane, 2-ship, 3-storage tank, 4-baseball diamond, 5-tennis court,
6-basketball court, 7-ground track field, 8-harbor, 9-bridge, 10-vehicle.

We keep class 1 (airplane) and emit normalised ``cls cx cy w h`` YOLO labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from .. import AIRPLANE_CLASS_ID

GT_DIRNAME = "ground truth"
IMG_DIRNAME = "positive image set"


def parse_annotation_line(line: str) -> tuple[int, int, int, int, int] | None:
    """Parse one ground-truth line into ``(x1, y1, x2, y2, class_id)``.

    Returns None for blank or malformed lines rather than raising, because the
    released dataset contains a handful of stray empty lines.
    """
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.replace("(", "").replace(")", "").split(",") if p.strip()]
    if len(parts) != 5:
        return None
    try:
        x1, y1, x2, y2, cls = (int(float(p)) for p in parts)
    except ValueError:
        return None
    return x1, y1, x2, y2, cls


def to_yolo_box(
    image_size: tuple[int, int], box: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    """Corner box in pixels -> normalised ``(cx, cy, w, h)``."""
    width, height = image_size
    x1, y1, x2, y2 = box
    # Guard against annotations whose corners are stored in the wrong order.
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def find_dataset_root(root: Path) -> Path:
    """Locate the directory that actually holds `ground truth/` and the image set.

    The Kaggle archive nests everything one level deep inside
    ``NWPU VHR-10 dataset/``; a manual unzip may not. Accept either.
    """
    root = Path(root)
    candidates = [root, *sorted(p for p in root.glob("*") if p.is_dir())]
    for candidate in candidates:
        if (candidate / GT_DIRNAME).is_dir() and (candidate / IMG_DIRNAME).is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{GT_DIRNAME}/' and '{IMG_DIRNAME}/' under {root}. "
        "Point --source at the extracted NWPU VHR-10 folder."
    )


def prepare_dataset(
    source: Path, dest: Path, class_id: int = AIRPLANE_CLASS_ID, overwrite: bool = False
) -> dict:
    """Build ``dest/{images,labels}`` from the NWPU VHR-10 release.

    Only images containing at least one instance of `class_id` are copied, and
    only that class's boxes are written. Returns a summary dict, also saved as
    ``dest/dataset_stats.json``.
    """
    source_root = find_dataset_root(Path(source))
    gt_dir = source_root / GT_DIRNAME
    img_dir = source_root / IMG_DIRNAME

    dest = Path(dest)
    if dest.exists() and overwrite:
        shutil.rmtree(dest)
    images_out = dest / "images"
    labels_out = dest / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes = 0
    n_skipped_lines = 0
    n_missing_images = 0

    for gt_path in sorted(gt_dir.glob("*.txt")):
        img_path = img_dir / f"{gt_path.stem}.jpg"
        if not img_path.exists():
            n_missing_images += 1
            continue

        with Image.open(img_path) as img:
            size = img.size  # (width, height)

        lines: list[str] = []
        for raw in gt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = parse_annotation_line(raw)
            if parsed is None:
                if raw.strip():
                    n_skipped_lines += 1
                continue
            *box, cls = parsed
            if cls != class_id:
                continue
            cx, cy, bw, bh = to_yolo_box(size, tuple(box))  # type: ignore[arg-type]
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                n_skipped_lines += 1
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue

        shutil.copy(img_path, images_out / img_path.name)
        (labels_out / f"{gt_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_images += 1
        n_boxes += len(lines)

    stats = {
        "source": str(source_root),
        "class_id": class_id,
        "class_name": "airplane" if class_id == AIRPLANE_CLASS_ID else f"class_{class_id}",
        "images": n_images,
        "boxes": n_boxes,
        "boxes_per_image": round(n_boxes / n_images, 2) if n_images else 0.0,
        "skipped_lines": n_skipped_lines,
        "missing_images": n_missing_images,
    }
    (dest / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="extracted NWPU VHR-10 folder")
    parser.add_argument("--dest", type=Path, default=Path("data/aircraft"))
    parser.add_argument(
        "--class-id", type=int, default=AIRPLANE_CLASS_ID, help="NWPU class id to keep (1=airplane)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = prepare_dataset(args.source, args.dest, args.class_id, args.overwrite)
    print(json.dumps(stats, indent=2))
    if stats["images"] == 0:
        raise SystemExit(f"No images found for class {args.class_id}.")


if __name__ == "__main__":
    main()
