"""Tests for the learning-rate schedule, seeding and config invariants.

The schedule is here because it is where a real bug lived: warmup and cosine
decay were once two schedulers attached to the same optimizer, stepped
alternately, which produced neither intended curve.
"""

import random
import unittest

import numpy as np
import torch

from aircraft_detector.config import Config
from aircraft_detector.models.yolo_tiny import STRIDE
from aircraft_detector.train import build_scheduler
from aircraft_detector.utils import set_seed


def lr_curve(epochs: int, warmup: int, lr: float = 1e-3) -> list[float]:
    """Learning rate actually applied at each epoch."""
    cfg = Config(epochs=epochs, warmup_epochs=warmup, lr=lr)
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=lr)
    scheduler = build_scheduler(optimizer, cfg)
    seen = []
    for _ in range(epochs):
        seen.append(optimizer.param_groups[0]["lr"])
        param.grad = torch.zeros_like(param)   # step the optimizer first, as torch expects
        optimizer.step()
        scheduler.step()
    return seen


class TestScheduler(unittest.TestCase):
    def test_warmup_ramps_linearly_from_a_fraction_of_the_base_rate(self):
        curve = lr_curve(epochs=100, warmup=5, lr=1e-3)
        self.assertAlmostEqual(curve[0], 1e-3 * 1 / 5, places=9)
        self.assertAlmostEqual(curve[4], 1e-3, places=9)
        warmup = curve[:5]
        self.assertEqual(warmup, sorted(warmup), "warmup must increase monotonically")

    def test_decay_is_monotonic_after_warmup(self):
        after = lr_curve(epochs=100, warmup=5)[5:]
        self.assertEqual(after, sorted(after, reverse=True), "cosine phase must not rise")

    def test_no_discontinuity_at_the_warmup_handover(self):
        """The old two-scheduler arrangement produced a jump exactly here."""
        curve = lr_curve(epochs=100, warmup=5, lr=1e-3)
        self.assertLess(abs(curve[5] - curve[4]) / 1e-3, 0.05)

    def test_final_rate_is_near_zero(self):
        curve = lr_curve(epochs=50, warmup=5, lr=1e-3)
        self.assertLess(curve[-1], 1e-3 * 0.01)

    def test_rate_never_exceeds_the_configured_base(self):
        self.assertLessEqual(max(lr_curve(epochs=60, warmup=5, lr=1e-3)), 1e-3 + 1e-12)

    def test_zero_warmup_is_treated_as_one_epoch(self):
        """`max(warmup, 1)` guards a division by zero."""
        curve = lr_curve(epochs=10, warmup=0, lr=1e-3)
        self.assertEqual(len(curve), 10)
        self.assertTrue(all(np.isfinite(curve)))


class TestSeeding(unittest.TestCase):
    def test_same_seed_reproduces_every_rng(self):
        set_seed(123)
        first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        set_seed(123)
        second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(first, second)

    def test_different_seeds_diverge(self):
        set_seed(1)
        a = float(torch.rand(1))
        set_seed(2)
        self.assertNotEqual(a, float(torch.rand(1)))


class TestConfigInvariants(unittest.TestCase):
    def test_grid_size_is_derived_from_image_size(self):
        self.assertEqual(Config(image_size=640).grid_size, 640 // STRIDE)
        self.assertEqual(Config(image_size=320).grid_size, 320 // STRIDE)

    def test_grid_size_cannot_be_set_into_an_inconsistent_state(self):
        with self.assertRaises(TypeError):
            Config(grid_size=99)

    def test_image_size_must_be_a_multiple_of_the_stride(self):
        with self.assertRaises(ValueError):
            Config(image_size=500)

    def test_split_fractions_must_leave_a_training_set(self):
        with self.assertRaises(ValueError):
            Config(val_fraction=0.6, test_fraction=0.5)

    def test_paths_are_coerced_from_strings(self):
        from pathlib import Path

        cfg = Config(dataset_dir="some/where", output_dir="runs/x")
        self.assertIsInstance(cfg.dataset_dir, Path)
        self.assertIsInstance(cfg.output_dir, Path)


if __name__ == "__main__":
    unittest.main()
