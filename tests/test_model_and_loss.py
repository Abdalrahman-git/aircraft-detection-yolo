import unittest

import torch

from aircraft_detector.losses import DetectionLoss, decode_grid, focal_bce
from aircraft_detector.models import YOLOTiny
from aircraft_detector.postprocess import decode_predictions


class TestModel(unittest.TestCase):
    def test_output_shape_matches_declared_grid(self):
        model = YOLOTiny()
        out = model(torch.zeros(2, 3, 320, 320))
        self.assertEqual(tuple(out.shape), (2, 10, 10, 5))
        self.assertEqual(YOLOTiny.grid_size(320), 10)

    def test_grid_size_helper_matches_forward_pass(self):
        """The original notebook discovered S with a dummy forward pass at runtime."""
        for size in (64, 320, 640):
            expected = YOLOTiny.grid_size(size)
            out = YOLOTiny()(torch.zeros(1, 3, size, size))
            self.assertEqual(out.shape[1], expected)

    def test_rejects_non_multiple_of_stride(self):
        with self.assertRaises(ValueError):
            YOLOTiny.grid_size(500)

    def test_outputs_are_probabilities(self):
        with torch.no_grad():
            out = YOLOTiny()(torch.rand(1, 3, 64, 64))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_objectness_starts_near_zero(self):
        """Biased init: >99% of cells are background, so start by predicting none."""
        model = YOLOTiny().eval()
        with torch.no_grad():
            out = model(torch.rand(4, 3, 64, 64))
        self.assertLess(float(out[..., 4].mean()), 0.1)

    def test_backward_pass_produces_gradients(self):
        model = YOLOTiny()
        out = model(torch.rand(1, 3, 64, 64))
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


class TestDecodeGrid(unittest.TestCase):
    def test_cell_offsets_become_image_coordinates(self):
        S = 4
        pred = torch.zeros(1, S, S, 5)
        pred[0, 1, 2, :4] = torch.tensor([0.5, 0.5, 0.2, 0.2])  # row 1, col 2
        decoded = decode_grid(pred, S)[0, 1, 2]
        self.assertAlmostEqual(float(decoded[0]), (2 + 0.5) / S, places=6)
        self.assertAlmostEqual(float(decoded[1]), (1 + 0.5) / S, places=6)


class TestFocalLoss(unittest.TestCase):
    def test_confident_correct_prediction_has_near_zero_loss(self):
        pred = torch.full((100,), 0.001)
        self.assertLess(float(focal_bce(pred, torch.zeros(100))), 1e-4)

    def test_confidently_wrong_prediction_is_penalised_more(self):
        target = torch.zeros(10)
        easy = focal_bce(torch.full((10,), 0.01), target)
        hard = focal_bce(torch.full((10,), 0.9), target)
        self.assertGreater(float(hard), float(easy))

    def test_gamma_down_weights_easy_examples(self):
        pred, target = torch.full((10,), 0.2), torch.zeros(10)
        self.assertLess(
            float(focal_bce(pred, target, gamma=2.0)),
            float(focal_bce(pred, target, gamma=0.0)),
        )

    def test_no_nan_at_the_probability_boundaries(self):
        for value in (0.0, 1.0):
            loss = focal_bce(torch.full((5,), value), torch.zeros(5))
            self.assertTrue(torch.isfinite(loss), f"non-finite loss at p={value}")


class TestDetectionLoss(unittest.TestCase):
    def setUp(self):
        self.S = 4
        self.criterion = DetectionLoss(self.S)

    def test_empty_target_still_produces_a_finite_loss(self):
        pred = torch.rand(2, self.S, self.S, 5)
        loss, parts = self.criterion(pred, torch.zeros(2, self.S, self.S, 5))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["box"], 0.0)

    def test_perfect_prediction_beats_a_wrong_one(self):
        target = torch.zeros(1, self.S, self.S, 5)
        target[0, 1, 1] = torch.tensor([0.5, 0.5, 0.2, 0.2, 1.0])

        perfect = target.clone()
        perfect[..., 4] = perfect[..., 4].clamp(min=1e-6)  # background stays ~0
        wrong = torch.zeros(1, self.S, self.S, 5)
        wrong[0, 1, 1] = torch.tensor([0.1, 0.9, 0.6, 0.6, 0.2])

        good, _ = self.criterion(perfect, target)
        bad, _ = self.criterion(wrong, target)
        self.assertLess(float(good), float(bad))

    def test_loss_is_differentiable(self):
        pred = torch.rand(1, self.S, self.S, 5, requires_grad=True)
        target = torch.zeros(1, self.S, self.S, 5)
        target[0, 0, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2, 1.0])
        loss, _ = self.criterion(pred, target)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())


class TestDecodePredictions(unittest.TestCase):
    def test_threshold_filters_low_confidence_cells(self):
        S = 4
        pred = torch.zeros(S, S, 5)
        pred[..., 2:4] = 0.1
        pred[1, 1, 4] = 0.9
        pred[2, 2, 4] = 0.2
        boxes, scores = decode_predictions(pred, conf_threshold=0.5)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)

    def test_returns_empty_when_nothing_passes(self):
        boxes, scores = decode_predictions(torch.zeros(4, 4, 5), conf_threshold=0.5)
        self.assertEqual(len(boxes), 0)
        self.assertEqual(len(scores), 0)

    def test_boxes_are_clamped_to_the_image(self):
        S = 4
        pred = torch.zeros(S, S, 5)
        pred[0, 0, :4] = torch.tensor([0.0, 0.0, 0.9, 0.9])  # extends past the top-left
        pred[0, 0, 4] = 0.99
        boxes, _ = decode_predictions(pred, conf_threshold=0.5)
        self.assertGreaterEqual(float(boxes.min()), 0.0)
        self.assertLessEqual(float(boxes.max()), 1.0)

    def test_rejects_a_batched_input(self):
        with self.assertRaises(ValueError):
            decode_predictions(torch.zeros(2, 4, 4, 5))


if __name__ == "__main__":
    unittest.main()
