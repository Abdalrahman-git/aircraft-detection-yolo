import unittest

import torch

from aircraft_detector.boxes import box_iou, complete_iou, cxcywh_to_xyxy, nms, xyxy_to_cxcywh


class TestConversions(unittest.TestCase):
    def test_roundtrip(self):
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.1, 0.9, 0.05, 0.05]])
        self.assertTrue(torch.allclose(xyxy_to_cxcywh(cxcywh_to_xyxy(boxes)), boxes, atol=1e-6))

    def test_known_corners(self):
        got = cxcywh_to_xyxy(torch.tensor([[0.5, 0.5, 0.2, 0.2]]))
        self.assertTrue(torch.allclose(got, torch.tensor([[0.4, 0.4, 0.6, 0.6]]), atol=1e-6))


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        self.assertAlmostEqual(float(box_iou(box, box)), 1.0, places=6)

    def test_disjoint_boxes(self):
        a = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
        b = torch.tensor([[0.9, 0.9, 1.0, 1.0]])
        self.assertAlmostEqual(float(box_iou(a, b)), 0.0, places=6)

    def test_half_overlap(self):
        a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        b = torch.tensor([[1.0, 0.0, 3.0, 2.0]])
        # intersection 2, union 6
        self.assertAlmostEqual(float(box_iou(a, b)), 1 / 3, places=6)

    def test_empty_inputs(self):
        a = torch.zeros((0, 4))
        b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        self.assertEqual(box_iou(a, b).shape, (0, 1))
        self.assertEqual(box_iou(b, a).shape, (1, 0))

    def test_ciou_equals_one_when_identical(self):
        box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        self.assertAlmostEqual(float(complete_iou(box, box)), 1.0, places=5)

    def test_ciou_penalises_distance_when_iou_is_zero(self):
        """The reason CIoU replaced plain IoU: it still ranks non-overlapping boxes."""
        target = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        near = torch.tensor([[0.7, 0.5, 0.1, 0.1]])
        far = torch.tensor([[0.95, 0.5, 0.1, 0.1]])
        self.assertAlmostEqual(float(box_iou(cxcywh_to_xyxy(near), cxcywh_to_xyxy(target))), 0.0)
        self.assertAlmostEqual(float(box_iou(cxcywh_to_xyxy(far), cxcywh_to_xyxy(target))), 0.0)
        self.assertGreater(float(complete_iou(near, target)), float(complete_iou(far, target)))


class TestNMS(unittest.TestCase):
    def test_suppresses_duplicates(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.2, 0.2],
                [0.01, 0.01, 0.21, 0.21],  # ~85% IoU with the first
                [0.8, 0.8, 1.0, 1.0],
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_threshold=0.45)
        self.assertEqual(keep.tolist(), [0, 2])

    def test_keeps_everything_when_threshold_is_high(self):
        boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.01, 0.01, 0.21, 0.21]])
        scores = torch.tensor([0.9, 0.8])
        self.assertEqual(len(nms(boxes, scores, iou_threshold=0.99)), 2)

    def test_returns_descending_score_order(self):
        boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]])
        keep = nms(boxes, torch.tensor([0.2, 0.9]))
        self.assertEqual(keep.tolist(), [1, 0])

    def test_empty(self):
        self.assertEqual(len(nms(torch.zeros((0, 4)), torch.zeros((0,)))), 0)


if __name__ == "__main__":
    unittest.main()
