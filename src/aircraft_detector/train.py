"""Training entry point: ``python -m aircraft_detector.train``."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config, add_config_arguments
from .data.dataset import build_splits, collate_fn
from .losses import DetectionLoss
from .metrics import evaluate_map, format_metrics
from .models import YOLOTiny
from .utils import get_device, seed_worker, set_seed


def build_scheduler(optimizer, cfg: Config):
    """Linear warmup into cosine decay, as a single schedule.

    The original notebook attached a ``LambdaLR`` *and* a ``CosineAnnealingLR``
    to the same optimizer and stepped whichever matched the epoch. Because
    ``LambdaLR`` rescales ``base_lr`` while ``CosineAnnealingLR`` tracks its own
    internal step count, the two fought over the learning rate and the schedule
    after epoch 5 did not match the intended cosine curve.
    """
    warmup = max(cfg.warmup_epochs, 1)

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(cfg.epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_loaders(cfg: Config) -> dict[str, DataLoader]:
    splits = build_splits(
        cfg.dataset_dir / "images",
        cfg.dataset_dir / "labels",
        cfg.val_fraction,
        cfg.test_fraction,
        cfg.seed,
        cfg.image_size,
        cfg.grid_size,
    )
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loaders = {}
    for name, dataset in splits.items():
        loaders[name] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(name == "train"),
            drop_last=(name == "train" and len(dataset) > cfg.batch_size),
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=generator if name == "train" else None,
        )
    return loaders


def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = get_device()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(cfg.output_dir / "config.yaml")

    loaders = make_loaders(cfg)
    sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"Device: {device}")
    print(f"Split sizes: {sizes}")

    model = YOLOTiny().to(device)
    expected_grid = YOLOTiny.grid_size(cfg.image_size)
    if expected_grid != cfg.grid_size:
        raise ValueError(
            f"config.grid_size={cfg.grid_size} but a {cfg.image_size}px input "
            f"yields a {expected_grid}x{expected_grid} grid"
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = DetectionLoss(
        cfg.grid_size,
        cfg.box_weight,
        cfg.obj_weight,
        cfg.noobj_weight,
        cfg.focal_gamma,
        cfg.focal_alpha,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)

    best_ap = -1.0
    best_path = cfg.output_dir / "best.pt"
    history: list[dict] = []
    started = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running = {"box": 0.0, "obj": 0.0, "noobj": 0.0, "total": 0.0}
        for images, targets, _ in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            loss, parts = criterion(model(images), targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            for key, value in parts.items():
                running[key] += value

        n_batches = max(len(loaders["train"]), 1)
        record = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v / n_batches for k, v in running.items()},
        }
        scheduler.step()

        is_eval_epoch = (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1
        if is_eval_epoch:
            metrics = evaluate_map(
                model, loaders["val"], device, cfg.conf_threshold, cfg.nms_iou_threshold
            )
            record.update({f"val_{k}": v for k, v in metrics.items()})
            print(
                f"Epoch {epoch + 1:3d} | loss {record['train_total']:.4f} | "
                f"{format_metrics(metrics)} | lr {record['lr']:.2e}"
            )
            if metrics["ap50"] > best_ap:
                best_ap = metrics["ap50"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "metrics": metrics,
                        "config": {
                            k: str(v) if isinstance(v, Path) else v
                            for k, v in asdict(cfg).items()
                        },
                    },
                    best_path,
                )
                print(f"  -> new best AP50 {best_ap:.3f}, checkpoint saved")
        history.append(record)

    (cfg.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Final report on the held-out test split, using the best checkpoint.
    summary = {"best_val_ap50": best_ap, "minutes": round((time.time() - started) / 60, 2)}
    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        test_metrics = evaluate_map(
            model, loaders["test"], device, cfg.conf_threshold, cfg.nms_iou_threshold
        )
        summary["test"] = test_metrics
        print(f"\nTest ({sizes['test']} images): {format_metrics(test_metrics)}")

    summary["splits"] = sizes
    summary["parameters"] = n_params
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Artifacts written to {cfg.output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the aircraft detector")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    add_config_arguments(
        parser,
        "dataset_dir",
        "output_dir",
        "epochs",
        "batch_size",
        "lr",
        "image_size",
        "grid_size",
        "num_workers",
        "eval_every",
        "seed",
        "conf_threshold",
    )
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    train(cfg.add_cli_overrides(args))


if __name__ == "__main__":
    main()
