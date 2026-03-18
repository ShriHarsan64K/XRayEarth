"""
XRayEarth — loss.py
Seeing through disaster with satellite vision.

Responsibilities:
    - Focal Loss implementation (primary contribution)
    - CrossEntropy baseline (for ablation comparison)
    - Loss switcher (config-driven: focal | cross_entropy)
    - Integration hook into Mask R-CNN's classification loss
    - Class weight computation from dataset statistics
    - Per-class loss tracking for WandB logging

Theory:
    Standard CrossEntropy treats all examples equally.
    In xBD, ~80% of buildings are undamaged → model learns
    to predict "no-damage" always and gets high accuracy
    but completely fails on destroyed buildings.

    Focal Loss adds a modulating factor (1 - pt)^gamma:
        FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    When gamma=2.0:
        - Easy examples (pt=0.9): weight = (0.1)^2 = 0.01
        - Hard examples (pt=0.1): weight = (0.9)^2 = 0.81

    This forces the model to focus on rare, hard instances
    (destroyed buildings) instead of easy majority class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from utils import console


# ═══════════════════════════════════════════════════════════
#  1. FOCAL LOSS
# ═══════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Originally proposed in:
        "Focal Loss for Dense Object Detection"
        Lin et al., 2017 (RetinaNet paper)

    Formula:
        FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    Where:
        pt      = model probability for the true class
        gamma   = focusing parameter (default 2.0)
                  higher gamma → more focus on hard examples
        alpha_t = class balancing weight

    Args:
        gamma:        Focusing parameter (default 2.0)
        alpha:        Class weight scalar or per-class tensor.
                      If float → applied to all non-background classes.
                      If Tensor[C] → per-class weights.
        reduction:    "mean" | "sum" | "none"
        ignore_index: Class index to ignore (-1 = none)
    """

    def __init__(
        self,
        gamma:        float                    = 2.0,
        alpha:        Optional[torch.Tensor]   = None,
        reduction:    str                      = "mean",
        ignore_index: int                      = -1,
    ):
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.reduction    = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        inputs:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            inputs:  [N, C] raw logits (before softmax)
            targets: [N]    ground truth class indices

        Returns:
            Scalar loss value
        """
        # Handle ignore_index
        if self.ignore_index >= 0:
            valid = targets != self.ignore_index
            inputs  = inputs[valid]
            targets = targets[valid]

        if inputs.numel() == 0:
            return inputs.sum() * 0.0  # zero loss, keeps grad

        # Compute standard cross entropy (log-softmax based)
        # ce_loss shape: [N]
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none"
        )

        # Compute pt = probability of true class
        # pt shape: [N]
        pt = torch.exp(-ce_loss)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1.0 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Per-class alpha: index by target class
                alpha_t = self.alpha.to(inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        # Final focal loss
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ═══════════════════════════════════════════════════════════
#  2. CROSS ENTROPY BASELINE
# ═══════════════════════════════════════════════════════════

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard CrossEntropy with optional class weights.
    Used as baseline (V1–V9) for ablation comparison.

    Args:
        weight:       Optional per-class weight tensor [C]
        ignore_index: Class index to ignore
        reduction:    "mean" | "sum" | "none"
    """

    def __init__(
        self,
        weight:       Optional[torch.Tensor] = None,
        ignore_index: int                    = -1,
        reduction:    str                    = "mean",
    ):
        super().__init__()
        self.weight       = weight
        self.ignore_index = ignore_index
        self.reduction    = reduction

    def forward(
        self,
        inputs:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs:  [N, C] logits
            targets: [N] ground truth indices
        """
        weight = self.weight.to(inputs.device) \
                 if self.weight is not None else None

        return F.cross_entropy(
            inputs,
            targets,
            weight       = weight,
            ignore_index = self.ignore_index if self.ignore_index >= 0 else -100,
            reduction    = self.reduction,
        )


# ═══════════════════════════════════════════════════════════
#  3. LOSS FACTORY
# ═══════════════════════════════════════════════════════════

def build_classification_loss(
    cfg,
    class_counts: Optional[Dict[int, int]] = None,
) -> nn.Module:
    """
    Build classification loss from config.

    Config options:
        loss.type:        "focal" | "cross_entropy"
        loss.focal_gamma: float (default 2.0)
        loss.focal_alpha: float (default 0.25)

    Args:
        cfg:          Full experiment config
        class_counts: Optional dict of {class_id: count}
                      Used to compute inverse-frequency weights.
                      If None, uniform weights used.

    Returns:
        Configured loss module (FocalLoss or WeightedCrossEntropyLoss)
    """
    loss_type = cfg.loss.type.lower()

    # Compute class weights from dataset statistics
    class_weights = None
    if class_counts is not None:
        class_weights = compute_class_weights(class_counts)
        console.log(
            f"[cyan]→[/cyan] Class weights: "
            + " | ".join(
                f"cls{k}={v:.3f}"
                for k, v in enumerate(class_weights.tolist())
            )
        )

    if loss_type == "focal":
        # Build alpha tensor from config + class weights
        alpha = _build_focal_alpha(cfg, class_weights)

        loss_fn = FocalLoss(
            gamma     = cfg.loss.focal_gamma,
            alpha     = alpha,
            reduction = "mean",
        )
        console.log(
            f"[green]✓[/green] Loss: [bold]Focal Loss[/bold] "
            f"(gamma={cfg.loss.focal_gamma}, alpha={cfg.loss.focal_alpha})"
        )

    elif loss_type == "cross_entropy":
        loss_fn = WeightedCrossEntropyLoss(
            weight    = class_weights,
            reduction = "mean",
        )
        console.log(
            f"[green]✓[/green] Loss: [bold]CrossEntropy[/bold]"
            + (" (weighted)" if class_weights is not None else "")
        )

    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. "
            f"Choose 'focal' or 'cross_entropy'."
        )

    return loss_fn


def _build_focal_alpha(
    cfg,
    class_weights: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Build the alpha tensor for Focal Loss.

    Strategy:
        - If class_weights provided: use inverse-frequency weights
        - Else: use scalar alpha from config (applied uniformly)

    Returns:
        Per-class alpha tensor [C] or None
    """
    if class_weights is not None:
        return class_weights
    else:
        # Scalar alpha — same for all classes
        return cfg.loss.focal_alpha


# ═══════════════════════════════════════════════════════════
#  4. CLASS WEIGHT COMPUTATION
# ═══════════════════════════════════════════════════════════

def compute_class_weights(
    class_counts: Dict[int, int],
    num_classes:  int = 5,        # 0=background + 4 damage classes
    method:       str = "inverse_frequency",
) -> torch.Tensor:
    """
    Compute per-class weights to handle class imbalance.

    Methods:
        inverse_frequency:
            weight_c = total_samples / (num_classes * count_c)
            Directly penalizes majority class.

        effective_samples (alternative):
            weight_c = (1 - beta) / (1 - beta^n_c)
            From "Class-Balanced Loss" (Cui et al., 2019)

    Args:
        class_counts: Dict mapping class_id → instance count
                      (1-indexed, background=0 excluded)
        num_classes:  Total number of classes including background
        method:       Weighting method

    Returns:
        Float tensor of shape [num_classes]
        Index 0 = background (weight=1.0)
        Index 1-4 = damage classes
    """
    weights = torch.ones(num_classes, dtype=torch.float32)

    # Only weight non-background classes
    damage_counts = {
        k: v for k, v in class_counts.items()
        if k > 0 and v > 0
    }

    if len(damage_counts) == 0:
        return weights

    total = sum(damage_counts.values())

    if method == "inverse_frequency":
        for cls_id, count in damage_counts.items():
            if cls_id < num_classes:
                weights[cls_id] = total / (len(damage_counts) * count)

    elif method == "effective_samples":
        beta = 0.9999
        for cls_id, count in damage_counts.items():
            if cls_id < num_classes:
                effective_n  = (1.0 - beta ** count) / (1.0 - beta)
                weights[cls_id] = 1.0 / effective_n

    # Normalize so mean weight = 1.0 (keeps loss scale stable)
    damage_weight_vals = weights[1:]  # exclude background
    damage_weight_vals = damage_weight_vals / damage_weight_vals.mean()
    weights[1:] = damage_weight_vals

    return weights


# ═══════════════════════════════════════════════════════════
#  5. MASK R-CNN LOSS INTEGRATION
# ═══════════════════════════════════════════════════════════

class XRayEarthLoss(nn.Module):
    """
    Complete loss wrapper for XRayEarth training.

    Mask R-CNN internally computes 5 losses:
        loss_classifier   ← we replace this with Focal/CE
        loss_box_reg      ← kept as standard smooth L1
        loss_mask         ← kept as standard binary CE
        loss_objectness   ← kept as standard BCE
        loss_rpn_box_reg  ← kept as standard smooth L1

    We only replace the ROI classification loss.
    The other losses are already computed by torchvision's
    Mask R-CNN and returned in the loss dict.

    Args:
        cfg:          Experiment config
        class_counts: Optional class distribution for weighting
    """

    def __init__(self, cfg, class_counts: Optional[Dict[int, int]] = None):
        super().__init__()
        self.cfg = cfg

        # Classification loss (the one we replace)
        self.cls_loss_fn = build_classification_loss(cfg, class_counts)

        # Loss weights for combining components
        self.loss_weights = {
            "loss_classifier":   1.0,
            "loss_box_reg":      1.0,
            "loss_mask":         1.0,
            "loss_objectness":   1.0,
            "loss_rpn_box_reg":  1.0,
        }

    def compute_total_loss(
        self,
        loss_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted total loss from Mask R-CNN loss dict.

        The loss_dict is returned directly by model(images, targets)
        in training mode. We scale each component by its weight.

        Args:
            loss_dict: Dict from Mask R-CNN forward pass

        Returns:
            (total_loss, scalar_loss_dict)
            total_loss:       Scalar tensor for backprop
            scalar_loss_dict: Python float dict for logging
        """
        total = torch.tensor(0.0, device=next(
            iter(loss_dict.values())
        ).device)

        scalar_dict = {}
        for key, loss_val in loss_dict.items():
            weight = self.loss_weights.get(key, 1.0)
            total  = total + weight * loss_val
            scalar_dict[key] = loss_val.item()

        scalar_dict["loss_total"] = total.item()
        return total, scalar_dict


# ═══════════════════════════════════════════════════════════
#  6. LOSS COMPARISON UTILITIES
# ═══════════════════════════════════════════════════════════

def compare_losses_on_batch(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    gamma:   float = 2.0,
    alpha:   float = 0.25,
) -> Dict[str, float]:
    """
    Compare Focal Loss vs CrossEntropy on the same batch.
    Used for analysis and visualization.

    Args:
        logits:  [N, C] raw prediction logits
        targets: [N]    ground truth labels
        gamma:   Focal loss gamma
        alpha:   Focal loss alpha

    Returns:
        Dict with both loss values and per-class breakdowns
    """
    ce_fn    = WeightedCrossEntropyLoss()
    focal_fn = FocalLoss(gamma=gamma, alpha=alpha)

    ce_loss    = ce_fn(logits, targets).item()
    focal_loss = focal_fn(logits, targets).item()

    # Per-class focal weights (for analysis)
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        true_probs = probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        focal_weights = (1 - true_probs) ** gamma

    return {
        "cross_entropy":        ce_loss,
        "focal_loss":           focal_loss,
        "focal_weight_mean":    focal_weights.mean().item(),
        "focal_weight_min":     focal_weights.min().item(),
        "focal_weight_max":     focal_weights.max().item(),
        "ratio_focal_to_ce":    focal_loss / (ce_loss + 1e-8),
    }


# ═══════════════════════════════════════════════════════════
#  7. QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from utils import load_config, get_project_root

    print("🧪 Testing loss.py...")

    root = get_project_root()

    # ── Test Focal Loss ────────────────────────────────────
    print("\n  Testing Focal Loss...")
    focal = FocalLoss(gamma=2.0, alpha=0.25)

    # Simulate imbalanced batch:
    # 80 easy "no-damage", 10 minor, 7 major, 3 destroyed
    N, C = 100, 5
    logits = torch.randn(N, C)

    targets = torch.cat([
        torch.zeros(80, dtype=torch.long),   # no-damage (majority)
        torch.ones(10,  dtype=torch.long),   # minor
        torch.full((7,), 2, dtype=torch.long),  # major
        torch.full((3,), 3, dtype=torch.long),  # destroyed (rare)
    ])

    fl = focal(logits, targets)
    assert fl.item() > 0
    print(f"  ✓ Focal Loss: {fl.item():.4f}")

    # ── Test CrossEntropy ──────────────────────────────────
    print("\n  Testing CrossEntropy...")
    ce = WeightedCrossEntropyLoss()
    cel = ce(logits, targets)
    assert cel.item() > 0
    print(f"  ✓ Cross Entropy: {cel.item():.4f}")

    # ── Verify focal < CE for easy majority class ──────────
    # On easy examples, focal should be lower
    easy_logits  = torch.zeros(10, C)
    easy_logits[:, 0] = 5.0              # very confident "no-damage"
    easy_targets = torch.zeros(10, dtype=torch.long)

    easy_fl  = focal(easy_logits, easy_targets).item()
    easy_ce  = ce(easy_logits, easy_targets).item()
    print(f"\n  Imbalance test (easy majority examples):")
    print(f"  ✓ CE loss:    {easy_ce:.6f}")
    print(f"  ✓ Focal loss: {easy_fl:.6f}  ← should be << CE")
    assert easy_fl < easy_ce, \
        "Focal loss should down-weight easy examples vs CE!"
    print(f"  ✓ Focal correctly down-weights easy examples ✅")

    # ── Test class weight computation ─────────────────────
    print("\n  Testing class weight computation...")
    # Simulate extreme imbalance
    counts = {1: 8000, 2: 1000, 3: 700, 4: 300}
    weights = compute_class_weights(counts)
    print(f"  ✓ Class weights: {[f'{w:.3f}' for w in weights.tolist()]}")
    assert weights[4] > weights[1], \
        "Destroyed class should have higher weight than no-damage!"
    print(f"  ✓ Rare class (destroyed) weight > majority class ✅")

    # ── Test loss comparison ───────────────────────────────
    print("\n  Testing loss comparison utility...")
    comparison = compare_losses_on_batch(logits, targets)
    for k, v in comparison.items():
        print(f"  {k}: {v:.4f}")

    # ── Test config-driven loss factory ───────────────────
    print("\n  Testing loss factory (V1 = CE, V10 = Focal)...")
    cfg_v1  = load_config(str(root / "configs" / "v1.yaml"))
    cfg_v10 = load_config(str(root / "configs" / "v10.yaml"))

    loss_v1  = build_classification_loss(cfg_v1)
    loss_v10 = build_classification_loss(cfg_v10)

    assert isinstance(loss_v1,  WeightedCrossEntropyLoss)
    assert isinstance(loss_v10, FocalLoss)
    print(f"  ✓ V1  loss type:  {type(loss_v1).__name__}")
    print(f"  ✓ V10 loss type:  {type(loss_v10).__name__}")

    print("\n✅ loss.py self-test passed!")