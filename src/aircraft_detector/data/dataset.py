"""Dataset, augmentation and deterministic train/val/test splitting.

Boxes are carried around as normalised ``(cx, cy, w, h)`` in [0, 1] throughout.

Each sample yields ``(image, target, boxes)``:

* ``image``  - float tensor ``[3, image_size, image_size]`` in [0, 1]
* ``target`` - float tensor ``[S, S, 5]``, the grid encoding used by the loss
* ``boxes``  - float tensor ``[N, 4]``, the *unencoded* ground truth

The third element matters: the grid encoding keeps at most one box per cell, so
scoring predictions against ``target`` would silently hide any object the
encoding dropped. Metrics are computed against ``boxes`` instead.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

# A box must retain at least this fraction of its area after a crop to be kept.
MIN_VISIBLE_AREA = 0.4


def load_boxes(label_path: Path) -> np.ndarray:
    """Read a YOLO label file into an ``[N, 4]`` array of ``(cx, cy, w, h)``."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append([float(v) for v in parts[1:]])  # column 0 is the class id (always 0)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def encode_target(boxes: np.ndarray, grid_size: int) -> tuple[torch.Tensor, int]:
    """Encode boxes into a ``[S, S, 5]`` grid target.

    Returns the target and the number of boxes dropped because their cell was
    already occupied (this head predicts one box per cell).
    """
    target = torch.zeros((grid_size, grid_size, 5), dtype=torch.float32)
    dropped = 0
    for cx, cy, bw, bh in boxes:
        col = min(int(cx * grid_size), grid_size - 1)
        row = min(int(cy * grid_size), grid_size - 1)
        if target[row, col, 4] == 1:
            dropped += 1
            continue
        target[row, col] = torch.tensor(
            [cx * grid_size - col, cy * grid_size - row, bw, bh, 1.0], dtype=torch.float32
        )
    return target, dropped


def _clip_boxes(boxes: np.ndarray) -> np.ndarray:
    """Clip centre-format boxes to the frame, dropping ones that mostly fell out.

    The original notebook clamped with ``w = min(w, 1 - x, x)``, which treats the
    full width as if it were a half-width and so halved every box near an edge.
    Clipping in corner space is the correct operation.
    """
    if len(boxes) == 0:
        return boxes
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2
    original_area = np.maximum((x2 - x1) * (y2 - y1), 1e-9)

    x1c, y1c = np.clip(x1, 0, 1), np.clip(y1, 0, 1)
    x2c, y2c = np.clip(x2, 0, 1), np.clip(y2, 0, 1)
    new_w, new_h = np.maximum(x2c - x1c, 0), np.maximum(y2c - y1c, 0)

    keep = (new_w * new_h) / original_area >= MIN_VISIBLE_AREA
    clipped = np.stack([(x1c + x2c) / 2, (y1c + y2c) / 2, new_w, new_h], axis=1)
    return clipped[keep].astype(np.float32)


class AircraftDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        files: list[str] | None = None,
        image_size: int = 640,
        grid_size: int = 20,
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment
        if files is None:
            files = sorted(p.name for p in self.image_dir.glob("*.jpg"))
        self.files = list(files)
        if not self.files:
            raise FileNotFoundError(f"No .jpg images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.files)

    # --- augmentation -------------------------------------------------

    def _random_crop(self, img: Image.Image, boxes: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        """Zoom in on a random sub-window. Never introduces border padding."""
        scale = random.uniform(0.8, 1.0)
        if scale >= 0.999:
            return img, boxes
        width, height = img.size
        crop_w, crop_h = int(width * scale), int(height * scale)
        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        img = img.crop((left, top, left + crop_w, top + crop_h))

        if len(boxes):
            u0, v0 = left / width, top / height
            cw, ch = crop_w / width, crop_h / height
            boxes = np.stack(
                [
                    (boxes[:, 0] - u0) / cw,
                    (boxes[:, 1] - v0) / ch,
                    boxes[:, 2] / cw,
                    boxes[:, 3] / ch,
                ],
                axis=1,
            )
            boxes = _clip_boxes(boxes)
        return img, boxes

    def _random_dihedral(
        self, img: Image.Image, boxes: np.ndarray
    ) -> tuple[Image.Image, np.ndarray]:
        """Flips and 90-degree rotations.

        Overhead imagery has no canonical "up", so the full 8-element dihedral
        group is label-preserving here - a much stronger prior than the
        horizontal flip alone, and it costs nothing.
        """
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                boxes[:, 0] = 1.0 - boxes[:, 0]
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if len(boxes):
                boxes[:, 1] = 1.0 - boxes[:, 1]
        if random.random() < 0.5:
            # Rotate 90 degrees counter-clockwise; the image is square by this point.
            img = img.transpose(Image.ROTATE_90)
            if len(boxes):
                cx, cy = boxes[:, 0].copy(), boxes[:, 1].copy()
                bw, bh = boxes[:, 2].copy(), boxes[:, 3].copy()
                boxes[:, 0], boxes[:, 1] = cy, 1.0 - cx
                boxes[:, 2], boxes[:, 3] = bh, bw
        return img, boxes

    def _color_jitter(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.5:  # hue roll, in HSV space
            hsv = np.array(img.convert("HSV"), dtype=np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-12, 12)) % 256
            img = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
        return img

    # --- item ---------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self.files[idx]
        img = Image.open(self.image_dir / name).convert("RGB")
        boxes = load_boxes(self.label_dir / f"{Path(name).stem}.txt")

        if self.augment:
            img, boxes = self._random_crop(img, boxes)

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.augment:
            img, boxes = self._random_dihedral(img, boxes)
            img = self._color_jitter(img)

        array = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        target, _ = encode_target(boxes, self.grid_size)
        return tensor, target, torch.from_numpy(boxes.astype(np.float32))


def collate_fn(batch):
    """Stack images and targets; keep the ragged ground-truth boxes as a list."""
    images, targets, boxes = zip(*batch, strict=True)
    return torch.stack(images), torch.stack(targets), list(boxes)


def build_splits(
    image_dir: str | Path,
    label_dir: str | Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    image_size: int,
    grid_size: int,
) -> dict[str, AircraftDataset]:
    """Split by *filename* so the three sets never share an image.

    Augmentation is enabled only on the training split, by constructing a
    separate dataset object per split. The original notebook instead flipped a
    flag on one shared dataset from inside ``__getitem__``, which leaked
    augmentation into validation whenever more than one worker was used.
    """
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    files = sorted(p.name for p in image_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}. Run `prepare` first.")

    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = round(n * test_fraction)
    n_val = round(n * val_fraction)
    if n - n_val - n_test < 1:
        raise ValueError(f"Splits leave no training images (n={n})")

    test_files = shuffled[:n_test]
    val_files = shuffled[n_test : n_test + n_val]
    train_files = shuffled[n_test + n_val :]

    def make(subset: list[str], augment: bool) -> AircraftDataset:
        return AircraftDataset(
            image_dir, label_dir, subset, image_size, grid_size, augment=augment
        )

    return {
        "train": make(train_files, augment=True),
        "val": make(val_files, augment=False),
        "test": make(test_files, augment=False),
    }
