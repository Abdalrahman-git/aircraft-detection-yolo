"""Box geometry shared by the loss, the NMS step and the metrics.

Two conventions are used and never mixed implicitly:

* ``cxcywh`` - normalised centre, width, height (dataset + model output)
* ``xyxy``   - normalised corners (IoU, NMS, matching)
"""

from __future__ import annotations

import math

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    width = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    height = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return width * height


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between ``[N, 4]`` and ``[M, 4]`` xyxy boxes -> ``[N, M]``."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / union.clamp(min=1e-9)


def complete_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise CIoU between two ``[N, 4]`` cxcywh tensors.

    CIoU extends IoU with a centre-distance term and an aspect-ratio term, so it
    still provides a gradient when boxes do not overlap at all. Plain IoU loss
    (used in the original notebook) is flat and gives no learning signal in that
    case, which is common early in training.
    """
    pred_xyxy = cxcywh_to_xyxy(pred)
    target_xyxy = cxcywh_to_xyxy(target)

    lt = torch.max(pred_xyxy[:, :2], target_xyxy[:, :2])
    rb = torch.min(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = (pred[:, 2] * pred[:, 3] + target[:, 2] * target[:, 3] - inter).clamp(min=1e-9)
    iou = inter / union

    # Smallest box enclosing both, used for the distance normaliser.
    enc_lt = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    enc_rb = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    diagonal = (enc_wh[:, 0] ** 2 + enc_wh[:, 1] ** 2).clamp(min=1e-9)
    centre_dist = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2

    v = (4 / math.pi**2) * (
        torch.atan(target[:, 2] / target[:, 3].clamp(min=1e-9))
        - torch.atan(pred[:, 2] / pred[:, 3].clamp(min=1e-9))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=1e-9)

    return iou - centre_dist / diagonal - alpha * v


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    """Greedy non-maximum suppression over ``[N, 4]`` xyxy boxes.

    Returns the indices to keep, in descending score order.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        best = order[0]
        keep.append(int(best))
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[best].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious < iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
