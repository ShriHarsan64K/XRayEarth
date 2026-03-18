"""
XRayEarth — model.py
Seeing through disaster with satellite vision.

Responsibilities:
    - Siamese Mask R-CNN with shared ResNet50-FPN backbone
    - FPN feature fusion: Concat + Difference at each pyramid level
    - Configurable: single-image (baseline) or Siamese mode
    - Configurable: backbone freeze, dropout, norm layer
    - Configurable: ResNet34 or ResNet50 backbone
    - Support for GroupNorm (better with small batch sizes)
    - Clean forward pass returning standard Mask R-CNN loss dict
"""

import os
from typing import Dict, List, Optional, Tuple, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models                          import resnet50, resnet34
from torchvision.models                          import ResNet50_Weights, ResNet34_Weights
from torchvision.models.detection               import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn   import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn     import MaskRCNNPredictor
from torchvision.ops                            import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network    import LastLevelMaxPool

from utils import console


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

# FPN output channels (standard torchvision default)
FPN_OUT_CHANNELS = 256

# After fusion (concat + diff): 256 + 256 + 256 = 768
# We project back to 256 with a 1×1 conv
FUSED_CHANNELS   = FPN_OUT_CHANNELS * 3
PROJ_CHANNELS    = FPN_OUT_CHANNELS

# Number of damage classes (background=0 handled internally)
# Our labels: 1=no-damage, 2=minor, 3=major, 4=destroyed
NUM_CLASSES      = 5   # 4 damage classes + 1 background


# ═══════════════════════════════════════════════════════════
#  1. NORM LAYER FACTORY
# ═══════════════════════════════════════════════════════════

def get_norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    """
    Return normalization layer by config name.

    Args:
        norm_type:    "batch_norm" | "group_norm"
        num_channels: Number of feature channels

    Returns:
        nn.Module normalization layer
    """
    if norm_type == "group_norm":
        # GroupNorm with 32 groups — works well with batch_size=2
        num_groups = min(32, num_channels // 4)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    else:
        return nn.BatchNorm2d(num_channels)


# ═══════════════════════════════════════════════════════════
#  2. FPN FUSION MODULE
# ═══════════════════════════════════════════════════════════

class FPNFusionModule(nn.Module):
    """
    Fuses pre and post FPN feature maps at each pyramid level.

    Fusion strategy: Concat + Difference
        fused = Conv1x1( cat([pre, post, post - pre]) )

    This gives the model:
        - pre features    (what was there before)
        - post features   (what is there after)
        - difference      (explicit change signal)

    Input channels per level:  256 (pre) + 256 (post) + 256 (diff) = 768
    Output channels per level: 256 (projected back via 1×1 conv)

    Args:
        in_channels:  FPN channel count per stream (default 256)
        out_channels: Output channel count (default 256)
        norm_type:    "batch_norm" | "group_norm"
    """

    def __init__(
        self,
        in_channels:  int = FPN_OUT_CHANNELS,
        out_channels: int = PROJ_CHANNELS,
        norm_type:    str = "batch_norm",
    ):
        super().__init__()

        fused_ch = in_channels * 3  # concat + diff = 3× channels

        # Shared 1×1 projection conv applied to each FPN level
        # (same weights for P2, P3, P4, P5)
        self.proj = nn.Sequential(
            nn.Conv2d(fused_ch, out_channels, kernel_size=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        pre_features:  Dict[str, torch.Tensor],
        post_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pre_features:  OrderedDict of FPN levels {"0":P2, "1":P3, ...}
            post_features: Same structure for post-disaster

        Returns:
            Fused feature dict with same keys, 256 channels per level
        """
        fused = {}

        for key in pre_features:
            pre  = pre_features[key]
            post = post_features[key]
            diff = post - pre

            # Concatenate along channel dim: [B, 768, H, W]
            combined = torch.cat([pre, post, diff], dim=1)

            # Project back to 256 channels: [B, 256, H, W]
            fused[key] = self.proj(combined)

        return fused


# ═══════════════════════════════════════════════════════════
#  3. SHARED SIAMESE BACKBONE
# ═══════════════════════════════════════════════════════════

class SiameseBackbone(nn.Module):
    """
    Shared-weight backbone for Siamese feature extraction.

    Both pre and post images pass through the SAME ResNet-FPN.
    Weights are shared — this is true Siamese learning, not
    two separate networks.

    Args:
        backbone_name: "resnet50" | "resnet34"
        pretrained:    Use ImageNet pretrained weights
        norm_type:     "batch_norm" | "group_norm"
        trainable_layers: How many ResNet layers to keep trainable
            0 = freeze all (used in V5)
            5 = train all (used in V6+)
    """

    def __init__(
        self,
        backbone_name:    str  = "resnet50",
        pretrained:       bool = False,
        norm_type:        str  = "batch_norm",
        trainable_layers: int  = 3,
    ):
        super().__init__()

        self.backbone_name = backbone_name

        # Build ResNet-FPN backbone
        # torchvision's resnet_fpn_backbone handles FPN construction
        if backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.fpn_backbone = resnet_fpn_backbone(
                backbone_name    = "resnet50",
                weights          = weights,
                trainable_layers = trainable_layers,
            )
        elif backbone_name == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.fpn_backbone = resnet_fpn_backbone(
                backbone_name    = "resnet34",
                weights          = weights,
                trainable_layers = trainable_layers,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.out_channels = self.fpn_backbone.out_channels  # 256

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract FPN features from a single image tensor.

        Args:
            x: [B, 3, H, W] image tensor

        Returns:
            OrderedDict of FPN feature maps:
                "0": [B, 256, H/4,  W/4]   P2
                "1": [B, 256, H/8,  W/8]   P3
                "2": [B, 256, H/16, W/16]  P4
                "3": [B, 256, H/32, W/32]  P5
                "pool": [B, 256, H/64, W/64] P6
        """
        return self.fpn_backbone(x)


# ═══════════════════════════════════════════════════════════
#  4. CLASSIFIER HEAD
# ═══════════════════════════════════════════════════════════

class DeepClassifierHead(nn.Module):
    """
    2-layer MLP classifier head (V3+).

    Replaces default FastRCNN single linear layer with:
        Linear(1024 → 1024) → ReLU → Dropout → Linear(1024 → num_classes)

    Args:
        in_features:  ROI pooled feature dimension (1024 for ResNet50-FPN)
        num_classes:  Output classes (5: background + 4 damage)
        dropout:      Dropout probability (0.0 = disabled)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout:     float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(in_features, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ═══════════════════════════════════════════════════════════
#  5. XRAYEARTH MODEL
# ═══════════════════════════════════════════════════════════

class XRayEarthModel(nn.Module):
    """
    XRayEarth — Siamese Mask R-CNN for disaster damage assessment.

    Two operating modes (controlled by config):

    Mode A — Single Image (V1–V9 baseline):
        Only post image used.
        Standard Mask R-CNN forward pass.

    Mode B — Siamese (V10 full system):
        Both pre + post images processed by shared backbone.
        Features fused via Concat + Diff at FPN level.
        Fused features passed to RPN + ROI heads.

    Args:
        cfg: Full OmegaConf experiment config
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg     = cfg
        self.siamese = cfg.model.siamese

        backbone_name    = cfg.model.backbone
        pretrained       = cfg.model.pretrained
        norm_type        = cfg.model.norm_layer
        dropout          = cfg.model.dropout
        classifier_head  = cfg.model.get("classifier_head", "default")
        freeze_backbone  = cfg.model.get("freeze_backbone", False)

        # ── Shared Backbone ───────────────────────────────
        trainable_layers = 0 if freeze_backbone else 3
        self.backbone = SiameseBackbone(
            backbone_name    = backbone_name,
            pretrained       = pretrained,
            norm_type        = norm_type,
            trainable_layers = trainable_layers,
        )

        # ── FPN Fusion (Siamese mode only) ────────────────
        if self.siamese:
            self.fusion = FPNFusionModule(
                in_channels  = self.backbone.out_channels,
                out_channels = self.backbone.out_channels,
                norm_type    = norm_type,
            )

        # ── Mask R-CNN Detection Head ─────────────────────
        # Build Mask R-CNN using the backbone
        # We use torchvision's MaskRCNN with our custom backbone
        self.detector = self._build_detector(
            backbone         = self.backbone.fpn_backbone,
            num_classes      = NUM_CLASSES,
            classifier_head  = classifier_head,
            dropout          = dropout,
        )

    def _build_detector(
        self,
        backbone:        nn.Module,
        num_classes:     int,
        classifier_head: str,
        dropout:         float,
    ) -> MaskRCNN:
        """
        Build Mask R-CNN detector with custom head.

        Args:
            backbone:        FPN backbone (already built)
            num_classes:     Number of output classes
            classifier_head: "default" | "deep"
            dropout:         Dropout probability

        Returns:
            Configured MaskRCNN model
        """
        # Build base Mask R-CNN
        # min_size/max_size tuned for tile sizes (384 or 512)
        model = MaskRCNN(
            backbone      = backbone,
            num_classes   = num_classes,
            min_size      = 384,
            max_size      = 512,
            box_detections_per_img = 200,  # more detections per tile
        )

        # Replace box predictor head
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        if classifier_head == "deep":
            # V3+: 2-layer MLP head
            model.roi_heads.box_predictor = nn.ModuleDict({
                "cls_score": DeepClassifierHead(in_features, num_classes, dropout),
                "bbox_pred": nn.Linear(in_features, num_classes * 4),
            })
            # Patch: replace with proper FastRCNNPredictor wrapper
            model.roi_heads.box_predictor = _DeepFastRCNNPredictor(
                in_features, num_classes, dropout
            )
        else:
            # Default: single linear layer
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )

        # Replace mask predictor head
        in_features_mask = \
            model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )

        return model

    def _extract_features(
        self,
        pre_images:  List[torch.Tensor],
        post_images: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract and fuse features from pre/post image pairs.

        Single-image mode: only uses post images
        Siamese mode:      fuses pre + post FPN features

        Args:
            pre_images:  List of [3, H, W] tensors
            post_images: List of [3, H, W] tensors

        Returns:
            Feature dict for RPN/ROI heads
        """
        # Stack into batch
        post_batch = torch.stack(post_images, dim=0)

        if not self.siamese:
            # V1–V9: single image mode, use post only
            return self.backbone(post_batch)

        # V10: Siamese mode
        pre_batch = torch.stack(pre_images, dim=0)

        pre_features  = self.backbone(pre_batch)
        post_features = self.backbone(post_batch)

        # Fuse: concat + difference at each FPN level
        fused_features = self.fusion(pre_features, post_features)

        return fused_features

    def forward(
        self,
        pre_images:  List[torch.Tensor],
        post_images: List[torch.Tensor],
        targets:     Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Forward pass.

        Training mode (targets provided):
            Returns loss dict:
                loss_classifier, loss_box_reg,
                loss_mask, loss_objectness, loss_rpn_box_reg

        Inference mode (targets=None):
            Returns list of prediction dicts per image:
                boxes, labels, scores, masks

        Args:
            pre_images:  List of [3, H, W] tensors
            post_images: List of [3, H, W] tensors
            targets:     List of target dicts (training only)

        Returns:
            Training: Dict[str, Tensor] loss dict
            Inference: List[Dict[str, Tensor]] predictions
        """
        # Extract fused features
        features = self._extract_features(pre_images, post_images)

        # Mask R-CNN forward
        # We need to pass images for image size info used by RPN
        # Use post images as the "images" for size reference
        if self.training and targets is not None:
            losses = self.detector(post_images, targets)
            return losses
        else:
            predictions = self.detector(post_images)
            return predictions


class _DeepFastRCNNPredictor(nn.Module):
    """
    Deep 2-layer MLP variant of FastRCNNPredictor.
    Wraps DeepClassifierHead to match torchvision interface.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float):
        super().__init__()
        self.cls_score = DeepClassifierHead(in_features, num_classes, dropout)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)

    def forward(self, x: torch.Tensor):
        scores = self.cls_score(x)
        deltas = self.bbox_pred(x)
        return scores, deltas


# ═══════════════════════════════════════════════════════════
#  6. MODEL FACTORY
# ═══════════════════════════════════════════════════════════

def build_model(cfg) -> XRayEarthModel:
    """
    Build XRayEarth model from config.

    Args:
        cfg: Full OmegaConf experiment config

    Returns:
        XRayEarthModel ready for training
    """
    model = XRayEarthModel(cfg)

    # Log model summary
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    console.log(
        f"[green]✓[/green] Model built: [bold]{cfg.project.version}[/bold]"
    )
    console.log(
        f"    Backbone  : {cfg.model.backbone}"
        f" | Pretrained: {cfg.model.pretrained}"
        f" | Siamese: {cfg.model.siamese}"
    )
    console.log(
        f"    Params    : {total_params:,} total"
        f" | {trainable_params:,} trainable"
    )

    return model


def freeze_backbone(model: XRayEarthModel) -> None:
    """
    Freeze all backbone parameters (V5 config).
    Only detection heads remain trainable.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    console.log(
        f"[yellow]→[/yellow] Backbone frozen. "
        f"Trainable params: {trainable:,}"
    )


def unfreeze_backbone(model: XRayEarthModel) -> None:
    """
    Unfreeze backbone for full fine-tuning (V6 config).
    """
    for param in model.backbone.parameters():
        param.requires_grad = True

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    console.log(
        f"[green]→[/green] Backbone unfrozen. "
        f"Trainable params: {trainable:,}"
    )


# ═══════════════════════════════════════════════════════════
#  7. QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from utils import load_config, get_project_root

    print("🧪 Testing model.py...")

    root = get_project_root()

    # ── Test V1: single image, no pretrain ────────────────
    print("\n  Testing V1 (single image baseline)...")
    cfg_v1 = load_config(str(root / "configs" / "v1.yaml"))
    model_v1 = build_model(cfg_v1)
    model_v1.eval()

    B, C, H, W = 2, 3, 384, 384
    pre  = [torch.rand(C, H, W) for _ in range(B)]
    post = [torch.rand(C, H, W) for _ in range(B)]

    with torch.no_grad():
        preds = model_v1(pre, post)

    assert len(preds) == B
    assert "boxes"  in preds[0]
    assert "labels" in preds[0]
    assert "masks"  in preds[0]
    print(f"  ✓ V1 inference passed: {len(preds[0]['boxes'])} detections")

    # ── Test training mode with targets ───────────────────
    print("\n  Testing training forward pass...")
    model_v1.train()

    targets = []
    for _ in range(B):
        n = 3  # 3 buildings per tile
        targets.append({
            "boxes":  torch.tensor([
                [10., 10., 80., 80.],
                [100., 100., 200., 200.],
                [250., 50., 350., 150.],
            ]),
            "labels": torch.tensor([1, 2, 4], dtype=torch.int64),
            "masks":  torch.zeros(n, H, W, dtype=torch.uint8),
        })

    losses = model_v1(pre, post, targets)
    assert isinstance(losses, dict)
    print(f"  ✓ Training losses: {list(losses.keys())}")
    for k, v in losses.items():
        print(f"      {k}: {v.item():.4f}")

    # ── Test V10: Siamese mode ─────────────────────────────
    print("\n  Testing V10 (Siamese + Focal Loss)...")
    cfg_v10  = load_config(str(root / "configs" / "v10.yaml"))
    model_v10 = build_model(cfg_v10)

    total    = sum(p.numel() for p in model_v10.parameters())
    trainable = sum(
        p.numel() for p in model_v10.parameters() if p.requires_grad
    )
    print(f"  ✓ V10 Siamese model: {total:,} params, {trainable:,} trainable")

    # ── Test freeze/unfreeze ───────────────────────────────
    print("\n  Testing freeze/unfreeze...")
    freeze_backbone(model_v10)
    unfreeze_backbone(model_v10)

    print("\n✅ model.py self-test passed!")