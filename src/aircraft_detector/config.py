"""Typed configuration, loadable from YAML and overridable from the CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # --- data ---
    dataset_dir: Path = Path("data/aircraft")
    image_size: int = 640
    grid_size: int = 20  # 640 / 32; must match the model stride (see models.yolo_tiny)

    # --- splits ---
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    # --- training ---
    epochs: int = 200
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 10.0
    num_workers: int = 0
    eval_every: int = 10

    # --- loss weights ---
    box_weight: float = 5.0
    obj_weight: float = 5.0
    noobj_weight: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # --- inference ---
    conf_threshold: float = 0.35
    nms_iou_threshold: float = 0.45

    # --- io ---
    output_dir: Path = Path("runs/train")

    _PATH_FIELDS = ("dataset_dir", "output_dir")

    def __post_init__(self) -> None:
        for name in self._PATH_FIELDS:
            setattr(self, name, Path(getattr(self, name)))
        if self.image_size % self.grid_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"grid_size ({self.grid_size})"
            )
        if not 0.0 <= self.val_fraction + self.test_fraction < 1.0:
            raise ValueError("val_fraction + test_fraction must be in [0, 1)")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def add_cli_overrides(self, args: argparse.Namespace) -> Config:
        """Apply any non-None argparse value whose name matches a config field."""
        known = {f.name for f in fields(self)}
        for key, value in vars(args).items():
            if value is not None and key in known:
                setattr(self, key, value)
        self.__post_init__()
        return self


def add_config_arguments(parser: argparse.ArgumentParser, *names: str) -> None:
    """Register `--field` overrides for the named Config fields, defaulting to None."""
    types: dict[str, Any] = {f.name: f.type for f in fields(Config)}
    casters = {"int": int, "float": float, "str": str, "Path": Path}
    for name in names:
        raw = types[name]
        raw = raw if isinstance(raw, str) else getattr(raw, "__name__", "str")
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=casters.get(raw, str),
            default=None,
            help=f"override config.{name}",
        )
