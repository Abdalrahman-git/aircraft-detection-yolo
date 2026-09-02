"""Detection metrics: precision/recall/F1 and average precision.

This module supplies ``evaluate_map``, which the original notebook called from
inside its training loop but never defined anywhere - training crashed with a
``NameError`` the first time it reached the evaluation epoch.

Two things are done deliberately here:

1. Metrics are computed against the *raw* ground-truth boxes, not against the
   grid-encoded target, so the one-box-per-cell encoding cannot hide a missed
   object and quietly inflate recall.
2. Average precision sweeps the full confidence range, while precision, recall
   and F1 are reported at the operating threshold you would actually deploy.
   Reporting only the latter, as the original did, makes runs with different
   confidence calibration incomparable.
"""

from __future__ import annotations

import numpy as np
import torch

from .boxes import box_iou, cxcywh_to_xyxy
from .postprocess import decode_predictions

# Detections below this score are never considered, even when sweeping for AP.
MIN_SCORE = 1e-3
COCO_IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))


def match_detections(
    det_boxes: torch.Tensor,
    det_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> tuple[np.ndarray, int]:
    """Greedily match detections (score order) to ground truth for one image.

    Returns a boolean array flagging each detection as a true positive, plus the
    number of ground-truth boxes. Each ground-truth box can be matched once;
    later duplicate detections of the same object count as false positives,
    which is what makes NMS quality visible in the score.
    """
    n_gt = int(gt_boxes.shape[0])
    n_det = int(det_boxes.shape[0])
    if n_det == 0:
        return np.zeros((0,), dtype=bool), n_gt
    if n_gt == 0:
        return np.zeros((n_det,), dtype=bool), 0

    order = det_scores.argsort(descending=True)
    ious = box_iou(det_boxes[order], gt_boxes).numpy()

    is_tp = np.zeros((n_det,), dtype=bool)
    claimed = np.zeros((n_gt,), dtype=bool)
    for rank in range(n_det):
        candidates = np.where(~claimed, ious[rank], -1.0)
        best = int(candidates.argmax())
        if candidates[best] >= iou_threshold:
            claimed[best] = True
            is_tp[rank] = True

    # Undo the score-ordering so the caller can align flags with scores.
    restored = np.zeros((n_det,), dtype=bool)
    restored[order.numpy()] = is_tp
    return restored, n_gt


def average_precision(
    all_scores: np.ndarray, all_tp: np.ndarray, total_gt: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """All-point interpolated AP (the VOC2010+ / COCO definition)."""
    if total_gt == 0:
        return float("nan"), np.array([]), np.array([])
    if all_scores.size == 0:
        return 0.0, np.array([0.0]), np.array([0.0])

    order = np.argsort(-all_scores)
    tp = all_tp[order].astype(np.float64)
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # Make precision monotonically decreasing, then integrate over recall.
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ap = float(np.sum(np.diff(mrec) * mpre[1:]))
    return ap, recall, precision


Detections = list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@torch.no_grad()
def collect_detections(
    model: torch.nn.Module,
    loader,
    device: torch.device | str,
    nms_iou_threshold: float = 0.45,
) -> Detections:
    """Run the model once over a loader, returning ``(boxes, scores, gt)`` per image.

    Detections are kept down to ``MIN_SCORE`` so a caller can re-score at any
    confidence threshold without another forward pass.
    """
    was_training = model.training
    model.eval()

    per_image: Detections = []
    for images, _targets, gt_boxes in loader:
        preds = model(images.to(device)).detach().cpu()
        for pred, gt in zip(preds, gt_boxes, strict=True):
            boxes, scores = decode_predictions(pred, MIN_SCORE, nms_iou_threshold)
            gt_xyxy = cxcywh_to_xyxy(gt).clamp(0, 1) if gt.numel() else torch.zeros((0, 4))
            per_image.append((boxes, scores, gt_xyxy))

    if was_training:
        model.train()
    return per_image


def score_detections(
    per_image: Detections,
    conf_threshold: float = 0.35,
    ap_iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS,
) -> dict[str, float]:
    """Score already-collected detections.

    Returns ``precision``, ``recall``, ``f1`` (at ``conf_threshold``, IoU 0.5),
    ``ap50``, ``ap50_95``, and the raw counts behind them.
    """
    total_gt = sum(int(gt.shape[0]) for _, _, gt in per_image)

    # --- AP across IoU thresholds ---
    ap_per_threshold: dict[float, float] = {}
    for iou_t in ap_iou_thresholds:
        scores_all, tp_all = [], []
        for boxes, scores, gt in per_image:
            is_tp, _ = match_detections(boxes, scores, gt, iou_t)
            scores_all.append(scores.numpy())
            tp_all.append(is_tp)
        ap, _, _ = average_precision(
            np.concatenate(scores_all) if scores_all else np.array([]),
            np.concatenate(tp_all) if tp_all else np.array([], dtype=bool),
            total_gt,
        )
        ap_per_threshold[float(iou_t)] = ap

    # --- P / R / F1 at the operating point, IoU 0.5 ---
    tp = fp = 0
    for boxes, scores, gt in per_image:
        keep = scores >= conf_threshold
        is_tp, _ = match_detections(boxes[keep], scores[keep], gt, 0.5)
        tp += int(is_tp.sum())
        fp += int((~is_tp).sum())
    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    valid_aps = [v for v in ap_per_threshold.values() if not np.isnan(v)]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": ap_per_threshold.get(0.5, float("nan")),
        "ap50_95": float(np.mean(valid_aps)) if valid_aps else float("nan"),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "num_ground_truth": total_gt,
        "conf_threshold": conf_threshold,
    }


def evaluate_map(
    model: torch.nn.Module,
    loader,
    device: torch.device | str,
    conf_threshold: float = 0.35,
    nms_iou_threshold: float = 0.45,
    ap_iou_thresholds: tuple[float, ...] = COCO_IOU_THRESHOLDS,
) -> dict[str, float]:
    """Evaluate a detector over a dataloader: collect once, then score."""
    per_image = collect_detections(model, loader, device, nms_iou_threshold)
    return score_detections(per_image, conf_threshold, ap_iou_thresholds)


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"P {metrics['precision']:.3f} | R {metrics['recall']:.3f} | "
        f"F1 {metrics['f1']:.3f} | AP50 {metrics['ap50']:.3f} | "
        f"AP50-95 {metrics['ap50_95']:.3f}"
    )
