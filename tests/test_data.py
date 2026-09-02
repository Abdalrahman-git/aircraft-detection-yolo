import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aircraft_detector.data.dataset import (
    AircraftDataset,
    _clip_boxes,
    build_splits,
    encode_target,
    load_boxes,
)
from aircraft_detector.data.prepare import parse_annotation_line, prepare_dataset, to_yolo_box


class TestAnnotationParsing(unittest.TestCase):
    def test_standard_line(self):
        self.assertEqual(parse_annotation_line("(563,478),(630,573),1"), (563, 478, 630, 573, 1))

    def test_line_with_spaces(self):
        self.assertEqual(parse_annotation_line(" (1, 2), (3, 4), 7 "), (1, 2, 3, 4, 7))

    def test_blank_and_malformed_lines_return_none(self):
        for bad in ["", "   ", "(1,2),(3,4)", "garbage", "(a,b),(c,d),1"]:
            self.assertIsNone(parse_annotation_line(bad), bad)

    def test_to_yolo_box_centres_and_normalises(self):
        cx, cy, w, h = to_yolo_box((100, 200), (10, 20, 30, 60))
        self.assertAlmostEqual(cx, 0.2)
        self.assertAlmostEqual(cy, 0.2)
        self.assertAlmostEqual(w, 0.2)
        self.assertAlmostEqual(h, 0.2)

    def test_to_yolo_box_handles_swapped_corners(self):
        self.assertEqual(
            to_yolo_box((100, 100), (30, 60, 10, 20)),
            to_yolo_box((100, 100), (10, 20, 30, 60)),
        )


class TestClipBoxes(unittest.TestCase):
    def test_fully_inside_box_is_untouched(self):
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
        self.assertTrue(np.allclose(_clip_boxes(boxes), boxes))

    def test_edge_box_keeps_its_visible_width(self):
        """The original code did `w = min(w, 1-x, x)`, halving boxes near an edge."""
        # Centre 0.05, width 0.1 -> spans exactly [0.0, 0.1], nothing to clip.
        boxes = np.array([[0.05, 0.5, 0.1, 0.1]], dtype=np.float32)
        clipped = _clip_boxes(boxes)
        self.assertEqual(len(clipped), 1)
        self.assertAlmostEqual(float(clipped[0][2]), 0.1, places=5)

    def test_mostly_outside_box_is_dropped(self):
        # Clipped in both axes to 0.12 x 0.12 of an original 0.2 x 0.2,
        # i.e. 36% of the area survives, under the 40% keep threshold.
        boxes = np.array([[0.02, 0.02, 0.2, 0.2]], dtype=np.float32)
        self.assertEqual(len(_clip_boxes(boxes)), 0)

    def test_box_just_above_the_visibility_threshold_is_kept(self):
        # Spans [-0.09, 0.11] in x only: 55% of the area survives.
        boxes = np.array([[0.01, 0.5, 0.2, 0.2]], dtype=np.float32)
        self.assertEqual(len(_clip_boxes(boxes)), 1)

    def test_partially_outside_box_is_trimmed(self):
        boxes = np.array([[0.05, 0.5, 0.2, 0.2]], dtype=np.float32)  # spans [-0.05, 0.15]
        clipped = _clip_boxes(boxes)
        self.assertEqual(len(clipped), 1)
        self.assertAlmostEqual(float(clipped[0][2]), 0.15, places=5)
        self.assertAlmostEqual(float(clipped[0][0]), 0.075, places=5)

    def test_empty_input(self):
        self.assertEqual(len(_clip_boxes(np.zeros((0, 4), dtype=np.float32))), 0)


class TestEncodeTarget(unittest.TestCase):
    def test_box_lands_in_the_right_cell(self):
        target, dropped = encode_target(np.array([[0.52, 0.28, 0.1, 0.2]]), grid_size=20)
        self.assertEqual(dropped, 0)
        row, col = 5, 10  # 0.28*20 = 5.6, 0.52*20 = 10.4
        self.assertEqual(float(target[row, col, 4]), 1.0)
        self.assertAlmostEqual(float(target[row, col, 0]), 0.4, places=4)
        self.assertAlmostEqual(float(target[row, col, 1]), 0.6, places=4)
        self.assertEqual(float(target[..., 4].sum()), 1.0)

    def test_second_box_in_the_same_cell_is_reported_as_dropped(self):
        boxes = np.array([[0.51, 0.51, 0.1, 0.1], [0.52, 0.52, 0.1, 0.1]])
        _, dropped = encode_target(boxes, grid_size=4)
        self.assertEqual(dropped, 1)

    def test_box_on_the_far_edge_does_not_overflow_the_grid(self):
        target, dropped = encode_target(np.array([[1.0, 1.0, 0.1, 0.1]]), grid_size=20)
        self.assertEqual(dropped, 0)
        self.assertEqual(float(target[19, 19, 4]), 1.0)


def _make_fake_nwpu(root: Path, n_images: int = 8) -> Path:
    """Build a miniature dataset in the on-disk NWPU layout."""
    base = root / "NWPU VHR-10 dataset"
    gt, img = base / "ground truth", base / "positive image set"
    gt.mkdir(parents=True)
    img.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_images):
        Image.fromarray(rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)).save(
            img / f"{i:03d}.jpg"
        )
        lines = ["(10,20),(30,50),1", "(40,40),(60,70),1"]
        if i % 2:  # some storage tanks that must be filtered out
            lines.append("(5,5),(15,15),3")
        (gt / f"{i:03d}.txt").write_text("\n".join(lines) + "\n\n")
    return base


class TestPrepare(unittest.TestCase):
    def test_keeps_only_the_airplane_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_fake_nwpu(tmp, n_images=6)
            stats = prepare_dataset(tmp, tmp / "out", class_id=1)
            self.assertEqual(stats["images"], 6)
            self.assertEqual(stats["boxes"], 12)  # 2 airplanes each, tanks excluded
            self.assertEqual(stats["class_name"], "airplane")

    def test_selecting_a_class_that_is_absent_yields_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_fake_nwpu(tmp, n_images=4)
            stats = prepare_dataset(tmp, tmp / "out", class_id=9)  # bridge
            self.assertEqual(stats["images"], 0)

    def test_written_labels_are_normalised(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_fake_nwpu(tmp, n_images=2)
            prepare_dataset(tmp, tmp / "out", class_id=1)
            boxes = load_boxes(tmp / "out" / "labels" / "000.txt")
            self.assertEqual(boxes.shape, (2, 4))
            self.assertTrue(((boxes >= 0) & (boxes <= 1)).all())


class TestDatasetAndSplits(unittest.TestCase):
    def _prepared(self, tmp: Path, n: int = 20) -> Path:
        _make_fake_nwpu(tmp, n_images=n)
        prepare_dataset(tmp, tmp / "out", class_id=1)
        return tmp / "out"

    def test_item_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._prepared(Path(tmp), n=4)
            ds = AircraftDataset(out / "images", out / "labels", image_size=64, grid_size=2)
            image, target, boxes = ds[0]
            self.assertEqual(tuple(image.shape), (3, 64, 64))
            self.assertEqual(tuple(target.shape), (2, 2, 5))
            self.assertEqual(boxes.shape[1], 4)
            self.assertTrue(float(image.min()) >= 0.0 and float(image.max()) <= 1.0)

    def test_augmentation_keeps_boxes_in_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._prepared(Path(tmp), n=4)
            ds = AircraftDataset(
                out / "images", out / "labels", image_size=64, grid_size=4, augment=True
            )
            for _ in range(40):
                _, _, boxes = ds[0]
                if boxes.numel():
                    self.assertTrue(bool(((boxes >= 0) & (boxes <= 1)).all()), boxes)

    def test_splits_are_disjoint_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._prepared(Path(tmp), n=20)
            kwargs = {
                "image_dir": out / "images",
                "label_dir": out / "labels",
                "val_fraction": 0.15,
                "test_fraction": 0.15,
                "seed": 42,
                "image_size": 64,
                "grid_size": 2,
            }
            a = build_splits(**kwargs)
            b = build_splits(**kwargs)
            names = {k: set(v.files) for k, v in a.items()}
            self.assertEqual(len(names["train"] | names["val"] | names["test"]), 20)
            self.assertFalse(names["train"] & names["val"])
            self.assertFalse(names["train"] & names["test"])
            self.assertFalse(names["val"] & names["test"])
            for split in ("train", "val", "test"):
                self.assertEqual(a[split].files, b[split].files, "splits must be reproducible")

    def test_only_the_training_split_augments(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._prepared(Path(tmp), n=20)
            splits = build_splits(out / "images", out / "labels", 0.15, 0.15, 42, 64, 2)
            self.assertTrue(splits["train"].augment)
            self.assertFalse(splits["val"].augment)
            self.assertFalse(splits["test"].augment)

    def test_validation_items_are_stable_across_reads(self):
        """Guards the bug where a shared augment flag leaked into validation."""
        with tempfile.TemporaryDirectory() as tmp:
            out = self._prepared(Path(tmp), n=20)
            splits = build_splits(out / "images", out / "labels", 0.15, 0.15, 42, 64, 2)
            first = splits["val"][0][0]
            _ = splits["train"][0]  # exercise the augmented split in between
            second = splits["val"][0][0]
            self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
