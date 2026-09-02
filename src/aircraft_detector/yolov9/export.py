"""Materialise our seeded splits in the directory layout Ultralytics expects.

Ultralytics discovers data from a `data.yaml` pointing at

    <root>/images/{train,val,test}/
    <root>/labels/{train,val,test}/

and will happily invent its own split if you let it. That would make any
comparison against the from-scratch baseline meaningless, so this module drives
the export from :func:`aircraft_detector.data.dataset.build_splits` - the exact
same seeded filename shuffle the baseline trains on.

Label files are copied unchanged: `prepare.py` already writes normalised
`cls cx cy w h`, which is precisely the format Ultralytics reads.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from ..config import Config, add_config_arguments
from ..data.dataset import build_splits

SPLITS = ("train", "val", "test")


def write_data_yaml(root: Path, class_name: str = "aircraft") -> Path:
    """Write the dataset descriptor Ultralytics reads."""
    root = Path(root).resolve()
    payload = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: class_name},
    }
    destination = root / "data.yaml"
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def export_for_ultralytics(cfg: Config, dest: Path, overwrite: bool = False) -> dict:
    """Copy each split's images and labels into the Ultralytics layout.

    Returns a summary including the per-split filenames, so the split used by
    both models is recorded on disk and auditable after the fact.
    """
    source_images = cfg.dataset_dir / "images"
    source_labels = cfg.dataset_dir / "labels"
    splits = build_splits(
        source_images,
        source_labels,
        cfg.val_fraction,
        cfg.test_fraction,
        cfg.seed,
        cfg.image_size,
        cfg.grid_size,
    )

    dest = Path(dest)
    if dest.exists() and overwrite:
        shutil.rmtree(dest)

    manifest: dict[str, list[str]] = {}
    for split in SPLITS:
        image_dir = dest / "images" / split
        label_dir = dest / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for name in splits[split].files:
            stem = Path(name).stem
            shutil.copy(source_images / name, image_dir / name)
            label_src = source_labels / f"{stem}.txt"
            if label_src.exists():
                shutil.copy(label_src, label_dir / f"{stem}.txt")
        manifest[split] = list(splits[split].files)

    data_yaml = write_data_yaml(dest)
    summary = {
        "data_yaml": str(data_yaml),
        "seed": cfg.seed,
        "counts": {split: len(names) for split, names in manifest.items()},
        "files": manifest,
    }
    (dest / "split_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dest", type=Path, default=Path("data/aircraft_yolo"))
    parser.add_argument("--overwrite", action="store_true")
    add_config_arguments(parser, "dataset_dir", "seed", "val_fraction", "test_fraction")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg.add_cli_overrides(args)

    summary = export_for_ultralytics(cfg, args.dest, args.overwrite)
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
