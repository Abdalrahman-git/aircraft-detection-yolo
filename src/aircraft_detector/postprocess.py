"""Turn raw grid predictions into a list of boxes."""

from __future__ import annotations

import torch

from .boxes import cxcywh_to_xyxy, nms
from .losses import decode_grid


def decode_predictions(
    pred: torch.Tensor,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one image's ``[S, S, 5]`` prediction.

    Returns ``(boxes_xyxy [N, 4], scores [N])`` after NMS, sorted by score.

    This is fully vectorised; the original notebook looped over every cell in
    Python, which dominated inference time on a 20x20 grid and would not scale.
    """
    if pred.dim() != 3 or pred.shape[-1] < 5:
        raise ValueError(f"expected [S, S, 5] prediction, got {tuple(pred.shape)}")

    grid_size = pred.shape[0]
    scores = pred[..., 4].reshape(-1)
    boxes = decode_grid(pred.unsqueeze(0), grid_size).reshape(-1, 4)

    keep = scores >= conf_threshold
    boxes, scores = boxes[keep], scores[keep]
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4)), scores.new_zeros((0,))

    boxes_xyxy = cxcywh_to_xyxy(boxes).clamp(0, 1)
    order = nms(boxes_xyxy, scores, iou_threshold)[:max_detections]
    return boxes_xyxy[order], scores[order]

