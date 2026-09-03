"""Tests for the comparison harness (no `ultralytics` needed)."""

import unittest

import torch

from aircraft_detector.benchmark import THRESHOLD_GRID, format_table, tune_threshold
from aircraft_detector.boxes import cxcywh_to_xyxy


def box(cx, cy, size=0.1):
    return cxcywh_to_xyxy(torch.tensor([[cx, cy, size, size]]))


class TestTuneThreshold(unittest.TestCase):
    def test_picks_a_threshold_that_excludes_a_low_scoring_false_positive(self):
        """One correct box at 0.9, one wrong box at 0.2: the cut belongs between."""
        gt = box(0.2, 0.2)
        dets = torch.cat([box(0.2, 0.2), box(0.8, 0.8)])
        per_image = [(dets, torch.tensor([0.9, 0.2]), gt)]
        chosen = tune_threshold(per_image)
        self.assertGreater(chosen, 0.2)
        self.assertLessEqual(chosen, 0.9)

    def test_keeps_a_low_threshold_when_every_detection_is_correct(self):
        gt = torch.cat([box(0.2, 0.2), box(0.8, 0.8)])
        per_image = [(gt, torch.tensor([0.3, 0.25]), gt)]
        self.assertLessEqual(tune_threshold(per_image), 0.25)

    def test_returns_a_value_from_the_grid(self):
        gt = box(0.5, 0.5)
        per_image = [(gt, torch.tensor([0.77]), gt)]
        self.assertIn(tune_threshold(per_image), THRESHOLD_GRID)

    def test_handles_a_model_that_detects_nothing(self):
        per_image = [(torch.zeros((0, 4)), torch.zeros((0,)), box(0.5, 0.5))]
        self.assertIn(tune_threshold(per_image), THRESHOLD_GRID)


class TestFormatTable(unittest.TestCase):
    def _entry(self, **metrics):
        base = {"ap50": 0.9, "ap50_95": 0.5, "precision": 0.8, "recall": 0.7, "f1": 0.75}
        base.update(metrics)
        return {"metrics": base, "parameters": 1234567, "conf_threshold": 0.65}

    def test_renders_one_column_per_model(self):
        table = format_table({"from-scratch": self._entry(), "yolov9s": self._entry(ap50=0.95)})
        lines = table.splitlines()
        self.assertIn("from-scratch", lines[0])
        self.assertIn("yolov9s", lines[0])
        # header + divider + 5 metrics + parameters + threshold
        self.assertEqual(len(lines), 9)
        self.assertTrue(all(line.startswith("|") and line.endswith("|") for line in lines))

    def test_reports_each_models_threshold(self):
        results = {"a": self._entry(), "b": self._entry()}
        results["b"]["conf_threshold"] = 0.30
        table = format_table(results)
        threshold_row = next(
            ln for ln in table.splitlines() if ln.startswith("| Conf. threshold")
        )
        self.assertIn("0.65", threshold_row)
        self.assertIn("0.30", threshold_row)

    def test_works_with_a_single_model(self):
        table = format_table({"from-scratch": self._entry()})
        self.assertIn("| Parameters | 1,234,567 |", table)


if __name__ == "__main__":
    unittest.main()
