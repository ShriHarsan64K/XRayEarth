"""
XRayEarth — eval.py
Seeing through disaster with satellite vision.

Responsibilities:
    - Mode 1: Per-building damage classification metrics
    - Mode 2: Change detection + classification metrics
    - Macro F1-score (PRIMARY metric for imbalanced classes)
    - mAP (mean Average Precision)
    - IoU per class
    - Precision / Recall per class
    - Confusion matrix generation
    - Full-image prediction reconstruction from tiles
    - WandB metric logging with visualizations

Why Macro F1 as primary metric:
    Accuracy fails on imbalanced data.
    Macro F1 computes F1 per class then averages — equally
    weights all classes regardless of frequency.
    A model that ignores "destroyed" class will score ~0.25
    on macro F1 even if accuracy is 95%.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

sys.path.insert(0, str(Path(__file__).parent))

from utils    import console, get_gpu_memory_mb
from tiling   import reconstruct_predictions, TileInfo
from dataset  import CLASS_NAMES, NUM_CLASSES


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

# Class indices (1-indexed, 0=background)
DAMAGE_CLASSES = [1, 2, 3, 4]
CLASS_LABELS   = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# IoU threshold for a detection to count as correct
IOU_THRESHOLD  = 0.5


# ═══════════════════════════════════════════════════════════
#  1. IoU UTILITIES
# ═══════════════════════════════════════════════════════════

def compute_iou_boxes(
    box_a: torch.Tensor,
    box_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        box_a: [N, 4] boxes [x1, y1, x2, y2]
        box_b: [M, 4] boxes [x1, y1, x2, y2]

    Returns:
        [N, M] IoU matrix
    """
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    inter_x1 = torch.max(box_a[:, None, 0], box_b[None, :, 0])
    inter_y1 = torch.max(box_a[:, None, 1], box_b[None, :, 1])
    inter_x2 = torch.min(box_a[:, None, 2], box_b[None, :, 2])
    inter_y2 = torch.min(box_a[:, None, 3], box_b[None, :, 3])

    inter_w    = (inter_x2 - inter_x1).clamp(min=0)
    inter_h    = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return inter_area / (union_area + 1e-6)


def compute_mask_iou(
    mask_pred: np.ndarray,
    mask_gt:   np.ndarray,
) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        mask_pred: HxW binary predicted mask
        mask_gt:   HxW binary ground truth mask

    Returns:
        Scalar IoU value
    """
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union        = np.logical_or(mask_pred,  mask_gt).sum()
    return float(intersection) / (float(union) + 1e-6)


# ═══════════════════════════════════════════════════════════
#  2. DETECTION MATCHING
# ═══════════════════════════════════════════════════════════

def match_predictions_to_gt(
    pred_boxes:  torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes:    torch.Tensor,
    gt_labels:   torch.Tensor,
    iou_threshold: float = IOU_THRESHOLD,
) -> Tuple[List[int], List[int]]:
    """
    Match predicted boxes to ground truth boxes using IoU.

    Strategy:
        - Sort predictions by score (high → low)
        - For each prediction, find best matching GT (IoU >= threshold)
        - Each GT can only be matched once

    Args:
        pred_boxes:    [N, 4] predicted boxes
        pred_labels:   [N]    predicted class labels
        pred_scores:   [N]    confidence scores
        gt_boxes:      [M, 4] ground truth boxes
        gt_labels:     [M]    ground truth labels
        iou_threshold: Minimum IoU for a match

    Returns:
        (matched_pred_labels, matched_gt_labels)
        Both are lists of integers for metric computation
    """
    matched_pred = []
    matched_gt   = []

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        # Unmatched GTs are false negatives
        for gt_lbl in gt_labels.tolist():
            matched_pred.append(0)       # background = miss
            matched_gt.append(gt_lbl)
        return matched_pred, matched_gt

    # Sort by confidence score descending
    sort_idx    = pred_scores.argsort(descending=True)
    pred_boxes  = pred_boxes[sort_idx]
    pred_labels = pred_labels[sort_idx]

    # IoU matrix [N, M]
    iou_matrix  = compute_iou_boxes(pred_boxes, gt_boxes)

    matched_gt_mask = torch.zeros(len(gt_boxes), dtype=torch.bool)

    for i in range(len(pred_boxes)):
        ious = iou_matrix[i]
        ious[matched_gt_mask] = 0  # don't rematch

        best_iou, best_j = ious.max(dim=0)

        if best_iou >= iou_threshold:
            matched_gt_mask[best_j] = True
            matched_pred.append(pred_labels[i].item())
            matched_gt.append(gt_labels[best_j].item())
        else:
            # False positive — predicted but no matching GT
            matched_pred.append(pred_labels[i].item())
            matched_gt.append(0)   # 0 = background (no GT)

    # Unmatched GTs → false negatives
    for j in range(len(gt_boxes)):
        if not matched_gt_mask[j]:
            matched_pred.append(0)
            matched_gt.append(gt_labels[j].item())

    return matched_pred, matched_gt


# ═══════════════════════════════════════════════════════════
#  3. METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════

def compute_metrics(
    all_pred_labels: List[int],
    all_gt_labels:   List[int],
) -> Dict[str, float]:
    """
    Compute all classification metrics from matched label lists.

    Args:
        all_pred_labels: Flattened list of predicted labels
        all_gt_labels:   Flattened list of ground truth labels

    Returns:
        Dict of metric name → float value
    """
    if len(all_gt_labels) == 0:
        return {"macro_f1": 0.0, "accuracy": 0.0}

    pred = np.array(all_pred_labels)
    gt   = np.array(all_gt_labels)

    # Only evaluate on building classes (1-4), ignore background (0)
    valid  = (gt > 0)
    pred_v = pred[valid]
    gt_v   = gt[valid]

    if len(gt_v) == 0:
        return {"macro_f1": 0.0, "accuracy": 0.0}

    # ── Primary metric: Macro F1 ──────────────────────────
    macro_f1 = f1_score(
        gt_v, pred_v,
        average      = "macro",
        labels       = DAMAGE_CLASSES,
        zero_division = 0,
    )

    # ── Per-class F1 ──────────────────────────────────────
    per_class_f1 = f1_score(
        gt_v, pred_v,
        average      = None,
        labels       = DAMAGE_CLASSES,
        zero_division = 0,
    )

    # ── Per-class Precision ───────────────────────────────
    per_class_prec = precision_score(
        gt_v, pred_v,
        average      = None,
        labels       = DAMAGE_CLASSES,
        zero_division = 0,
    )

    # ── Per-class Recall ──────────────────────────────────
    per_class_rec = recall_score(
        gt_v, pred_v,
        average      = None,
        labels       = DAMAGE_CLASSES,
        zero_division = 0,
    )

    # ── Overall Accuracy ──────────────────────────────────
    accuracy = (pred_v == gt_v).mean()

    # ── Build metrics dict ────────────────────────────────
    metrics = {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
    }

    for i, cls_name in enumerate(CLASS_LABELS):
        metrics[f"f1_{cls_name}"]        = float(per_class_f1[i])
        metrics[f"precision_{cls_name}"] = float(per_class_prec[i])
        metrics[f"recall_{cls_name}"]    = float(per_class_rec[i])

    return metrics


def compute_map(
    pred_boxes:  torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes:    torch.Tensor,
    gt_labels:   torch.Tensor,
    num_classes: int = NUM_CLASSES,
    iou_threshold: float = IOU_THRESHOLD,
) -> float:
    """
    Compute mean Average Precision (mAP) at given IoU threshold.

    Args:
        pred_boxes:    [N, 4]
        pred_scores:   [N]
        pred_labels:   [N]
        gt_boxes:      [M, 4]
        gt_labels:     [M]
        num_classes:   Total number of classes
        iou_threshold: IoU threshold for a TP

    Returns:
        Scalar mAP value
    """
    aps = []

    for cls in range(1, num_classes):
        pred_mask = pred_labels == cls
        gt_mask   = gt_labels   == cls

        cls_pred_boxes  = pred_boxes[pred_mask]
        cls_pred_scores = pred_scores[pred_mask]
        cls_gt_boxes    = gt_boxes[gt_mask]

        if len(cls_gt_boxes) == 0:
            continue

        if len(cls_pred_boxes) == 0:
            aps.append(0.0)
            continue

        # Sort by score descending
        sort_idx        = cls_pred_scores.argsort(descending=True)
        cls_pred_boxes  = cls_pred_boxes[sort_idx]
        cls_pred_scores = cls_pred_scores[sort_idx]

        iou_matrix   = compute_iou_boxes(cls_pred_boxes, cls_gt_boxes)
        matched_gt   = torch.zeros(len(cls_gt_boxes), dtype=torch.bool)

        tp = torch.zeros(len(cls_pred_boxes))
        fp = torch.zeros(len(cls_pred_boxes))

        for i in range(len(cls_pred_boxes)):
            ious = iou_matrix[i].clone()
            ious[matched_gt] = 0

            best_iou, best_j = ious.max(dim=0)

            if best_iou >= iou_threshold:
                tp[i]            = 1
                matched_gt[best_j] = True
            else:
                fp[i] = 1

        # Precision-recall curve
        tp_cum   = tp.cumsum(0)
        fp_cum   = fp.cumsum(0)
        recalls  = tp_cum / (len(cls_gt_boxes) + 1e-6)
        precs    = tp_cum / (tp_cum + fp_cum + 1e-6)

        # AP via trapezoidal integration
        ap = torch.trapz(precs, recalls).item()
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


# ═══════════════════════════════════════════════════════════
#  4. CONFUSION MATRIX VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrix(
    all_pred: List[int],
    all_gt:   List[int],
    version:  str,
    save_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Generate and save a confusion matrix heatmap.

    Args:
        all_pred: Predicted labels
        all_gt:   Ground truth labels
        version:  Experiment version string
        save_dir: Directory to save the plot

    Returns:
        Path to saved image or None
    """
    # Filter to building classes only
    pred = np.array(all_pred)
    gt   = np.array(all_gt)
    valid = gt > 0
    pred, gt = pred[valid], gt[valid]

    if len(gt) == 0:
        return None

    cm = confusion_matrix(gt, pred, labels=DAMAGE_CLASSES)

    # Normalize by row (true class)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm  = cm_norm / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        ax=axes[0],
    )
    axes[0].set_title(f"{version} — Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        ax=axes[1], vmin=0, vmax=1,
    )
    axes[1].set_title(f"{version} — Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = str(Path(save_dir) / f"{version}_confusion_matrix.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path

    plt.close()
    return None


# ═══════════════════════════════════════════════════════════
#  5. EVALUATION LOOP
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    cfg,
    epoch:      int  = 0,
    smoke_test: bool = False,
) -> Dict[str, float]:
    """
    Run full evaluation on validation/test set.

    Computes both Mode 1 and Mode 2 metrics:
        Mode 1: Per-building damage classification
        Mode 2: Change detection (damaged vs undamaged)

    Args:
        model:      XRayEarthModel in eval mode
        dataloader: Val/test DataLoader
        device:     Compute device
        cfg:        Experiment config
        epoch:      Current epoch (for logging)
        smoke_test: If True, evaluate only 2 batches

    Returns:
        Dict of metric name → float value
    """
    model.eval()

    # Accumulators for Mode 1
    all_pred_labels_m1: List[int] = []
    all_gt_labels_m1:   List[int] = []

    # Accumulators for Mode 2 (binary: damaged vs undamaged)
    all_pred_labels_m2: List[int] = []
    all_gt_labels_m2:   List[int] = []

    # For mAP computation
    all_pred_boxes:  List[torch.Tensor] = []
    all_pred_scores: List[torch.Tensor] = []
    all_pred_lbls:   List[torch.Tensor] = []
    all_gt_boxes:    List[torch.Tensor] = []
    all_gt_lbls:     List[torch.Tensor] = []

    pbar = tqdm(
        dataloader,
        desc      = f"Epoch {epoch:03d} [eval ]",
        leave     = False,
        dynamic_ncols = True,
    )

    for batch_idx, (pre_imgs, post_imgs, targets, tile_infos) in enumerate(pbar):

        # Smoke test: stop early
        if smoke_test and batch_idx >= 2:
            break

        # Move to device
        pre_imgs  = [img.to(device)  for img  in pre_imgs]
        post_imgs = [img.to(device)  for img  in post_imgs]

        # ── Inference ─────────────────────────────────────
        predictions = model(pre_imgs, post_imgs, targets=None)

        # ── Process each image in batch ───────────────────
        for pred, target in zip(predictions, targets):

            pred_boxes  = pred["boxes"].cpu()
            pred_labels = pred["labels"].cpu()
            pred_scores = pred["scores"].cpu()

            gt_boxes  = target["boxes"]
            gt_labels = target["labels"]

            # ── Mode 1: Damage classification ─────────────
            pred_m1, gt_m1 = match_predictions_to_gt(
                pred_boxes, pred_labels, pred_scores,
                gt_boxes,   gt_labels,
                iou_threshold = cfg.evaluation.iou_threshold,
            )
            all_pred_labels_m1.extend(pred_m1)
            all_gt_labels_m1.extend(gt_m1)

            # ── Mode 2: Change detection (binary) ─────────
            # Remap: no-damage(1) → 0, any damage(2,3,4) → 1
            pred_m2 = [1 if lbl > 1 else 0 for lbl in pred_m1]
            gt_m2   = [1 if lbl > 1 else 0 for lbl in gt_m1]
            all_pred_labels_m2.extend(pred_m2)
            all_gt_labels_m2.extend(gt_m2)

            # ── mAP accumulators ──────────────────────────
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                all_pred_lbls.append(pred_labels)
                all_gt_boxes.append(gt_boxes)
                all_gt_lbls.append(gt_labels)

        pbar.set_postfix({"gpu": f"{get_gpu_memory_mb():.0f}MB"})

    # ── Compute Metrics ───────────────────────────────────
    metrics = {}

    # Mode 1 metrics
    if cfg.evaluation.mode1 and len(all_gt_labels_m1) > 0:
        m1_metrics = compute_metrics(all_pred_labels_m1, all_gt_labels_m1)
        for k, v in m1_metrics.items():
            metrics[f"m1_{k}"] = v

        # Primary metric is Mode 1 macro F1
        metrics["macro_f1"] = m1_metrics["macro_f1"]

    # Mode 2 metrics (binary change detection)
    if cfg.evaluation.mode2 and len(all_gt_labels_m2) > 0:
        # Binary F1 for change detection
        pred_m2 = np.array(all_pred_labels_m2)
        gt_m2   = np.array(all_gt_labels_m2)
        valid   = gt_m2 >= 0

        binary_f1 = f1_score(
            gt_m2[valid], pred_m2[valid],
            average       = "binary",
            zero_division = 0,
        )
        binary_prec = precision_score(
            gt_m2[valid], pred_m2[valid],
            zero_division = 0,
        )
        binary_rec = recall_score(
            gt_m2[valid], pred_m2[valid],
            zero_division = 0,
        )

        metrics["m2_binary_f1"]        = float(binary_f1)
        metrics["m2_binary_precision"] = float(binary_prec)
        metrics["m2_binary_recall"]    = float(binary_rec)

    # mAP
    if len(all_pred_boxes) > 0:
        cat_pred_boxes  = torch.cat(all_pred_boxes,  dim=0)
        cat_pred_scores = torch.cat(all_pred_scores, dim=0)
        cat_pred_lbls   = torch.cat(all_pred_lbls,   dim=0)
        cat_gt_boxes    = torch.cat(all_gt_boxes,    dim=0)
        cat_gt_lbls     = torch.cat(all_gt_lbls,     dim=0)

        map_score = compute_map(
            cat_pred_boxes, cat_pred_scores, cat_pred_lbls,
            cat_gt_boxes,   cat_gt_lbls,
        )
        metrics["mAP"] = map_score

    # Default fallback
    if "macro_f1" not in metrics:
        metrics["macro_f1"] = 0.0

    return metrics


# ═══════════════════════════════════════════════════════════
#  6. FULL EVALUATION REPORT
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate_eval_report(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    cfg,
    save_dir:   str,
) -> Dict[str, float]:
    """
    Generate comprehensive evaluation report with:
        - Full metrics table
        - Confusion matrix plots
        - Per-class analysis
        - WandB logging

    Used after training completes for final evaluation.

    Args:
        model:      Trained XRayEarthModel
        dataloader: Test DataLoader
        device:     Compute device
        cfg:        Experiment config
        save_dir:   Directory to save plots

    Returns:
        Full metrics dict
    """
    console.rule(
        f"[bold blue]Final Evaluation — {cfg.project.version}[/bold blue]"
    )

    model.eval()

    all_pred_m1: List[int] = []
    all_gt_m1:   List[int] = []

    for pre_imgs, post_imgs, targets, _ in tqdm(
        dataloader, desc="Final evaluation"
    ):
        pre_imgs  = [img.to(device) for img in pre_imgs]
        post_imgs = [img.to(device) for img in post_imgs]

        predictions = model(pre_imgs, post_imgs)

        for pred, target in zip(predictions, targets):
            pred_m1, gt_m1 = match_predictions_to_gt(
                pred["boxes"].cpu(),
                pred["labels"].cpu(),
                pred["scores"].cpu(),
                target["boxes"],
                target["labels"],
            )
            all_pred_m1.extend(pred_m1)
            all_gt_m1.extend(gt_m1)

    # Compute metrics
    metrics = compute_metrics(all_pred_m1, all_gt_m1)

    # Classification report
    pred = np.array(all_pred_m1)
    gt   = np.array(all_gt_m1)
    valid = gt > 0

    report = classification_report(
        gt[valid], pred[valid],
        labels       = DAMAGE_CLASSES,
        target_names = CLASS_LABELS,
        zero_division = 0,
    )

    console.print(f"\n[bold]Classification Report:[/bold]\n{report}")

    # Confusion matrix
    cm_path = plot_confusion_matrix(
        all_pred_m1, all_gt_m1,
        version  = cfg.project.version,
        save_dir = save_dir,
    )

    # Log to WandB
    try:
        wandb.log({
            "final/macro_f1": metrics["macro_f1"],
            "final/accuracy": metrics.get("accuracy", 0),
            "final/confusion_matrix": wandb.Image(cm_path) if cm_path else None,
        })
    except Exception:
        pass  # WandB may not be active

    console.log(
        f"[bold green]★ Final macro_f1 = "
        f"{metrics['macro_f1']:.4f}[/bold green]"
    )

    return metrics


# ═══════════════════════════════════════════════════════════
#  7. QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Testing eval.py...")

    # ── Test IoU computation ───────────────────────────────
    box_a = torch.tensor([[0., 0., 10., 10.]])
    box_b = torch.tensor([[5., 5., 15., 15.]])
    iou   = compute_iou_boxes(box_a, box_b)
    expected = 25.0 / (100 + 100 - 25)
    assert abs(iou[0, 0].item() - expected) < 1e-4
    print(f"  ✓ IoU computation: {iou[0,0]:.4f} (expected {expected:.4f})")

    # ── Test matching ──────────────────────────────────────
    pred_boxes  = torch.tensor([[0., 0., 10., 10.], [50., 50., 60., 60.]])
    pred_labels = torch.tensor([1, 2])
    pred_scores = torch.tensor([0.9, 0.8])
    gt_boxes    = torch.tensor([[1., 1., 11., 11.]])
    gt_labels   = torch.tensor([1])

    matched_pred, matched_gt = match_predictions_to_gt(
        pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels, iou_threshold=0.5,
    )
    print(f"  ✓ Matching: pred={matched_pred}, gt={matched_gt}")

    # ── Test metrics computation ───────────────────────────
    # Simulate: model mostly gets no-damage right
    # but misses all destroyed cases
    np.random.seed(42)
    n = 200
    gt_labels_sim = np.concatenate([
        np.ones(160, dtype=int),           # 160 no-damage
        np.full(20, 2, dtype=int),         # 20 minor
        np.full(15, 3, dtype=int),         # 15 major
        np.full(5,  4, dtype=int),         # 5 destroyed
    ])
    # Bad model: always predicts no-damage
    bad_pred = np.ones(n, dtype=int)
    bad_metrics = compute_metrics(bad_pred.tolist(), gt_labels_sim.tolist())
    print(f"\n  Bad model (always predicts no-damage):")
    print(f"  ✓ Accuracy:  {bad_metrics['accuracy']:.4f}  ← looks good!")
    print(f"  ✓ Macro F1:  {bad_metrics['macro_f1']:.4f}  ← reveals the problem!")
    assert bad_metrics["accuracy"] > 0.7
    assert bad_metrics["macro_f1"] < 0.4
    print(f"  ✓ Macro F1 correctly penalizes class-ignorant model ✅")

    # Good model: gets all classes right
    good_pred    = gt_labels_sim.copy()
    good_metrics = compute_metrics(good_pred.tolist(), gt_labels_sim.tolist())
    print(f"\n  Perfect model:")
    print(f"  ✓ Accuracy:  {good_metrics['accuracy']:.4f}")
    print(f"  ✓ Macro F1:  {good_metrics['macro_f1']:.4f}")
    assert good_metrics["macro_f1"] > 0.99

    # ── Test mAP ──────────────────────────────────────────
    pred_b = torch.tensor([[0.,0.,10.,10.], [20.,20.,30.,30.]])
    pred_s = torch.tensor([0.9, 0.8])
    pred_l = torch.tensor([1, 2])
    gt_b   = torch.tensor([[1.,1.,11.,11.]])
    gt_l   = torch.tensor([1])

    map_val = compute_map(pred_b, pred_s, pred_l, gt_b, gt_l)
    print(f"\n  ✓ mAP computation: {map_val:.4f}")

    print("\n✅ eval.py self-test passed!")