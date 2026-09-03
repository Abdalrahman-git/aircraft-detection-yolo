"""The YOLOv9 comparison is only meaningful if both models see the same split.

These tests need no `ultralytics` install: the export step is pure file layout
plus our own seeded splitter.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from aircraft_detector.config import Config
from aircraft_detector.data.dataset import build_splits, load_boxes
from aircraft_detector.data.prepare import prepare_dataset
from aircraft_detector.yolov9.export import SPLITS, export_for_ultralytics, write_data_yaml


def _make_fake_nwpu(root: Path, n_images: int = 20) -> Path:
    base = root / "NWPU VHR-10 dataset"
    gt, img = base / "ground truth", base / "positive image set"
    gt.mkdir(parents=True)
    img.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_images):
        Image.fromarray(rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)).save(
            img / f"{i:03d}.jpg"
        )
        (gt / f"{i:03d}.txt").write_text("(10,20),(30,50),1\n(40,40),(60,70),1\n", encoding="utf-8")
    return base


class TestExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        _make_fake_nwpu(tmp, n_images=20)
        prepare_dataset(tmp, tmp / "prepared", class_id=1)
        self.tmp = tmp
        self.cfg = Config(
            dataset_dir=tmp / "prepared",
            val_fraction=0.15,
            test_fraction=0.15,
            seed=42,
            image_size=640,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_exported_splits_match_the_baseline_splits_exactly(self):
        """The whole benchmark rests on this: same seed, same files, same split."""
        dest = self.tmp / "yolo"
        export_for_ultralytics(self.cfg, dest)

        baseline = build_splits(
            self.cfg.dataset_dir / "images",
            self.cfg.dataset_dir / "labels",
            self.cfg.val_fraction,
            self.cfg.test_fraction,
            self.cfg.seed,
            self.cfg.image_size,
            self.cfg.grid_size,
        )
        for split in SPLITS:
            exported = sorted(p.name for p in (dest / "images" / split).glob("*.jpg"))
            self.assertEqual(exported, sorted(baseline[split].files), f"{split} split diverged")

    def test_every_image_has_its_label(self):
        dest = self.tmp / "yolo"
        export_for_ultralytics(self.cfg, dest)
        for split in SPLITS:
            images = {p.stem for p in (dest / "images" / split).glob("*.jpg")}
            labels = {p.stem for p in (dest / "labels" / split).glob("*.txt")}
            self.assertEqual(images, labels, f"{split} images and labels disagree")

    def test_splits_are_disjoint_after_export(self):
        dest = self.tmp / "yolo"
        export_for_ultralytics(self.cfg, dest)
        seen: dict[str, str] = {}
        for split in SPLITS:
            for path in (dest / "images" / split).glob("*.jpg"):
                self.assertNotIn(
                    path.name, seen, f"{path.name} in both {seen.get(path.name)} and {split}"
                )
                seen[path.name] = split
        self.assertEqual(len(seen), 20)

    def test_labels_are_copied_unchanged(self):
        """Ultralytics reads the same normalised format prepare.py writes."""
        dest = self.tmp / "yolo"
        export_for_ultralytics(self.cfg, dest)
        for split in SPLITS:
            for label in (dest / "labels" / split).glob("*.txt"):
                original = load_boxes(self.cfg.dataset_dir / "labels" / label.name)
                copied = load_boxes(label)
                self.assertTrue(np.allclose(original, copied))
                for line in label.read_text(encoding="utf-8").splitlines():
                    self.assertEqual(line.split()[0], "0", "single-class id must be 0")

    def test_data_yaml_is_valid_and_single_class(self):
        dest = self.tmp / "yolo"
        export_for_ultralytics(self.cfg, dest)
        payload = yaml.safe_load((dest / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(payload["names"], {0: "aircraft"})
        for split in SPLITS:
            self.assertEqual(payload[split], f"images/{split}")
            self.assertTrue((Path(payload["path"]) / payload[split]).is_dir())

    def test_manifest_records_the_split(self):
        dest = self.tmp / "yolo"
        summary = export_for_ultralytics(self.cfg, dest)
        self.assertTrue((dest / "split_manifest.json").exists())
        self.assertEqual(sum(summary["counts"].values()), 20)
        self.assertEqual(summary["seed"], 42)

    def test_overwrite_removes_a_stale_split(self):
        dest = self.tmp / "yolo"
        export_for_ultralytics(self.cfg, dest)
        stale = dest / "images" / "train" / "stale.jpg"
        stale.write_bytes(b"not an image")
        export_for_ultralytics(self.cfg, dest, overwrite=True)
        self.assertFalse(stale.exists())

    def test_write_data_yaml_uses_an_absolute_path(self):
        dest = self.tmp / "yolo_only_yaml"
        dest.mkdir()
        payload = yaml.safe_load(write_data_yaml(dest).read_text(encoding="utf-8"))
        self.assertTrue(Path(payload["path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
