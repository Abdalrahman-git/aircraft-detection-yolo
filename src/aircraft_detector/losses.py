"""Detection loss: focal objectness + CIoU box regression.

Roughly 99.7% of the 400 grid cells in an image are background. Weighting them
equally with the handful of positive cells makes the model collapse to
predicting nothing, or - if the background weight is dropped - to firing
everywhere. Focal loss down-weights the easy background cells so the hard ones
still contribute.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .boxes import complete_iou


def focal_bce(
    pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25
) -> torch.Tensor:
    """Focal binary cross-entropy on probabilities (not logits)."""
    pred = pred.clamp(min=1e-7, max=1 - 1e-7)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    p_t = torch.where(target == 1, pred, 1 - pred)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def decode_grid(pred: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Turn per-cell offsets into whole-image ``cxcywh`` boxes.

    ``pred`` is ``[..., S, S, >=4]`` with channels ``(tx, ty, w, h, ...)``.
    """
    device = pred.device
    cols = torch.arange(grid_size, device=device).view(1, 1, grid_size)
    rows = torch.arange(grid_size, device=device).view(1, grid_size, 1)
    cx = (cols + pred[..., 0]) / grid_size
    cy = (rows + pred[..., 1]) / grid_size
    return torch.stack([cx, cy, pred[..., 2], pred[..., 3]], dim=-1)


class DetectionLoss:
    def __init__(
        self,
        grid_size: int,
        box_weight: float = 5.0,
        obj_weight: float = 5.0,
        noobj_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        self.grid_size = grid_size
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        obj_mask = target[..., 4] == 1
        noobj_mask = ~obj_mask

        obj_scores = pred[..., 4][obj_mask]
        noobj_scores = pred[..., 4][noobj_mask]

        if obj_mask.any():
            obj_loss = F.binary_cross_entropy(
                obj_scores.clamp(1e-7, 1 - 1e-7), torch.ones_like(obj_scores)
            )
        else:
            obj_loss = pred.new_zeros(())

        noobj_loss = (
            focal_bce(
                noobj_scores,
                torch.zeros_like(noobj_scores),
                self.focal_gamma,
                self.focal_alpha,
            )
            if noobj_mask.any()
            else pred.new_zeros(())
        )

        if obj_mask.any():
            pred_boxes = decode_grid(pred, self.grid_size)[obj_mask]
            true_boxes = decode_grid(target, self.grid_size)[obj_mask]
            box_loss = (1 - complete_iou(pred_boxes, true_boxes)).mean()
        else:
            box_loss = pred.new_zeros(())

        total = (
            self.box_weight * box_loss
            + self.obj_weight * obj_loss
            + self.noobj_weight * noobj_loss
        )
        parts = {
            "box": box_loss.detach().item(),
            "obj": obj_loss.detach().item(),
            "noobj": noobj_loss.detach().item(),
            "total": total.detach().item(),
        }
        return total, parts
