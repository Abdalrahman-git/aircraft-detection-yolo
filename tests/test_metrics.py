"""Tests for the metrics that the original notebook was missing entirely."""

import unittest

import numpy as np
import torch

from aircraft_detector.boxes import cxcywh_to_xyxy
from aircraft_detector.metrics import average_precision, match_detections, score_detections


def box(cx, cy, size=0.1):
    return cxcywh_to_xyxy(torch.tensor([[cx, cy, size, size]]))


class TestMatching(unittest.TestCase):
    def test_perfect_detection_is_a_true_positive(self):
        gt = box(0.5, 0.5)
        is_tp, n_gt = match_detections(gt, torch.tensor([0.9]), gt, 0.5)
        self.assertEqual(n_gt, 1)
        self.assertTrue(is_tp.all())

    def test_far_detection_is_a_false_positive(self):
        is_tp, _ = match_detections(box(0.9, 0.9), torch.tensor([0.9]), box(0.1, 0.1), 0.5)
        self.assertFalse(is_tp.any())

    def test_duplicate_detections_count_once(self):
        """Two boxes on one object: the higher-scoring one wins, the other is an FP."""
        gt = box(0.5, 0.5)
        dets = torch.cat([box(0.5, 0.5), box(0.505, 0.5)])
        is_tp, n_gt = match_detections(dets, torch.tensor([0.9, 0.8]), gt, 0.5)
        self.assertEqual(n_gt, 1)
        self.assertEqual(int(is_tp.sum()), 1)
        self.assertTrue(is_tp[0], "the highest-scoring detection should claim the object")

    def test_no_detections(self):
        is_tp, n_gt = match_detections(torch.zeros((0, 4)), torch.zeros((0,)), box(0.5, 0.5), 0.5)
        self.assertEqual(is_tp.shape, (0,))
        self.assertEqual(n_gt, 1)

    def test_no_ground_truth_makes_everything_a_false_positive(self):
        is_tp, n_gt = match_detections(box(0.5, 0.5), torch.tensor([0.9]), torch.zeros((0, 4)), 0.5)
        self.assertEqual(n_gt, 0)
        self.assertFalse(is_tp.any())

    def test_flags_align_with_input_order_not_score_order(self):
        """Matching runs in score order but must return flags in input order."""
        gt = box(0.5, 0.5)
        dets = torch.cat([box(0.9, 0.9), box(0.5, 0.5)])  # low score first, correct box second
        is_tp, _ = match_detections(dets, torch.tensor([0.2, 0.99]), gt, 0.5)
        self.assertEqual(is_tp.tolist(), [False, True])


class TestAveragePrecision(unittest.TestCase):
    def test_perfect_ranking_gives_ap_one(self):
        scores = np.array([0.9, 0.8, 0.7])
        tp = np.array([True, True, True])
        ap, _, _ = average_precision(scores, tp, total_gt=3)
        self.assertAlmostEqual(ap, 1.0, places=6)

    def test_no_detections_gives_ap_zero(self):
        ap, _, _ = average_precision(np.array([]), np.array([], dtype=bool), total_gt=5)
        self.assertAlmostEqual(ap, 0.0, places=6)

    def test_half_recall_caps_ap(self):
        # Two of four objects found, both ranked first: precision 1 up to recall 0.5.
        ap, _, _ = average_precision(
            np.array([0.9, 0.8]), np.array([True, True]), total_gt=4
        )
        self.assertAlmostEqual(ap, 0.5, places=6)

    def test_false_positive_ahead_of_true_positive_lowers_ap(self):
        good, _, _ = average_precision(np.array([0.9, 0.8]), np.array([True, False]), total_gt=1)
        bad, _, _ = average_precision(np.array([0.9, 0.8]), np.array([False, True]), total_gt=1)
        self.assertGreater(good, bad)

    def test_ap_is_nan_without_ground_truth(self):
        ap, _, _ = average_precision(np.array([0.9]), np.array([False]), total_gt=0)
        self.assertTrue(np.isnan(ap))


class TestScoreDetections(unittest.TestCase):
    def _one_image(self, det_boxes, det_scores, gt_boxes):
        return [(det_boxes, torch.tensor(det_scores), gt_boxes)]

    def test_perfect_detection_scores_one(self):
        gt = box(0.5, 0.5)
        result = score_detections(self._one_image(gt, [0.9], gt), conf_threshold=0.35)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["ap50"], 1.0, places=5)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_negatives"], 0)

    def test_detection_below_threshold_counts_as_a_miss(self):
        """AP still sees the low-scoring detection; P/R/F1 at the operating point do not."""
        gt = box(0.5, 0.5)
        result = score_detections(self._one_image(gt, [0.1], gt), conf_threshold=0.35)
        self.assertEqual(result["true_positives"], 0)
        self.assertEqual(result["false_negatives"], 1)
        self.assertAlmostEqual(result["recall"], 0.0)
        self.assertAlmostEqual(result["ap50"], 1.0, places=5)

    def test_missed_object_lowers_recall_but_not_precision(self):
        gt = torch.cat([box(0.2, 0.2), box(0.8, 0.8)])
        result = score_detections(
            self._one_image(box(0.2, 0.2), [0.9], gt), conf_threshold=0.35
        )
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertEqual(result["num_ground_truth"], 2)

    def test_empty_evaluation_set_is_handled(self):
        result = score_detections([], conf_threshold=0.35)
        self.assertEqual(result["num_ground_truth"], 0)
        self.assertAlmostEqual(result["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
