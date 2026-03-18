"""
XRayEarth — dataset.py
Seeing through disaster with satellite vision.

Responsibilities:
    - Parse xBD JSON annotation files (polygon → mask/box/label)
    - Load pre/post disaster image pairs
    - Apply tiling pipeline with hybrid caching
    - Return PyTorch tensors in Mask R-CNN target format
    - Support both Mode 1 (damage classification) and
      Mode 2 (change detection) evaluation
    - Albumentations augmentation pipeline
"""

import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from shapely.geometry import Polygon
from shapely.errors import ShapelyError
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tiling import (
    TileCache,
    TileInfo,
    image_to_tiles,
    generate_tile_coords,
)
from utils import (
    get_config_hash,
    console,
)


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

# xBD damage label mapping → integer class
DAMAGE_LABEL_MAP = {
    "no-damage":        0,
    "minor-damage":     1,
    "major-damage":     2,
    "destroyed":        3,
    "un-classified":    0,   # treat as no-damage
}

# Human readable class names (index → name)
CLASS_NAMES = {
    0: "no-damage",
    1: "minor-damage",
    2: "major-damage",
    3: "destroyed",
}

NUM_CLASSES = 4   # background is handled by Mask R-CNN internally
                  # our labels are 1-indexed in targets (0 = background)


# ═══════════════════════════════════════════════════════════
#  1. xBD ANNOTATION PARSER
# ═══════════════════════════════════════════════════════════

def parse_xbd_annotation(
    label_path: str,
    image_h:    int,
    image_w:    int,
) -> Dict[str, np.ndarray]:
    """
    Parse a single xBD JSON annotation file.

    xBD annotation structure:
    {
        "features": {
            "xy": [
                {
                    "properties": {
                        "uid":         "...",
                        "subtype":     "no-damage" | "minor-damage" |
                                       "major-damage" | "destroyed",
                        "feature_type": "building"
                    },
                    "wkt": "POLYGON ((x1 y1, x2 y2, ...))"
                },
                ...
            ]
        }
    }

    Args:
        label_path: Path to xBD JSON annotation file
        image_h:    Image height (for mask rasterization)
        image_w:    Image width

    Returns:
        Dict with keys:
            boxes   : Nx4 float32 array [xmin, ymin, xmax, ymax]
            masks   : NxHxW uint8 binary mask array
            labels  : N int64 array (1-indexed: 1=no-damage ... 4=destroyed)
            uids    : List of building UIDs
    """
    with open(label_path, "r") as f:
        data = json.load(f)

    features = data.get("features", {}).get("xy", [])

    boxes  = []
    masks  = []
    labels = []
    uids   = []

    for feature in features:
        props        = feature.get("properties", {})
        feature_type = props.get("feature_type", "")

        # Only process buildings
        if feature_type != "building":
            continue

        uid     = props.get("uid", "unknown")
        subtype = props.get("subtype", "no-damage").lower().strip()

        # Map damage label → integer (1-indexed for Mask R-CNN)
        damage_class = DAMAGE_LABEL_MAP.get(subtype, 0) + 1  # +1: 0=background

        # Parse WKT polygon
        wkt = feature.get("wkt", "")
        if not wkt:
            continue

        try:
            polygon = parse_wkt_polygon(wkt)
            if polygon is None or polygon.is_empty:
                continue

            # Get polygon exterior coordinates
            coords = np.array(polygon.exterior.coords, dtype=np.float32)

            # Rasterize polygon to binary mask
            mask = rasterize_polygon(coords, image_h, image_w)
            if mask.sum() == 0:
                continue

            # Compute bounding box from mask
            ys, xs = np.where(mask)
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

            # Skip degenerate boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([float(xmin), float(ymin),
                          float(xmax), float(ymax)])
            masks.append(mask)
            labels.append(damage_class)
            uids.append(uid)

        except (ShapelyError, ValueError, Exception) as e:
            warnings.warn(f"Skipping feature {uid}: {e}")
            continue

    if len(boxes) == 0:
        return {
            "boxes":  np.zeros((0, 4),        dtype=np.float32),
            "masks":  np.zeros((0, image_h, image_w), dtype=np.uint8),
            "labels": np.zeros(0,             dtype=np.int64),
            "uids":   [],
        }

    return {
        "boxes":  np.array(boxes,  dtype=np.float32),
        "masks":  np.stack(masks,  axis=0).astype(np.uint8),
        "labels": np.array(labels, dtype=np.int64),
        "uids":   uids,
    }


def parse_wkt_polygon(wkt: str) -> Optional[Polygon]:
    """
    Parse a WKT POLYGON string into a Shapely Polygon.

    Handles:
        - Standard WKT: POLYGON ((x1 y1, x2 y2, ...))
        - Multipolygon: returns largest polygon

    Args:
        wkt: WKT geometry string

    Returns:
        Shapely Polygon or None if invalid
    """
    from shapely import wkt as shapely_wkt
    from shapely.geometry import MultiPolygon

    try:
        geom = shapely_wkt.loads(wkt)
    except Exception:
        return None

    if isinstance(geom, MultiPolygon):
        # Return largest polygon by area
        geom = max(geom.geoms, key=lambda p: p.area)

    if not isinstance(geom, Polygon):
        return None

    return geom


def rasterize_polygon(
    coords:  np.ndarray,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """
    Rasterize polygon coordinates to a binary mask.

    Args:
        coords:  Nx2 float array of (x, y) coordinates
        image_h: Output mask height
        image_w: Output mask width

    Returns:
        HxW binary uint8 mask
    """
    mask = np.zeros((image_h, image_w), dtype=np.uint8)

    # Convert to integer pixel coords
    pts = coords[:, :2].astype(np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Clamp to image bounds
    pts[:, :, 0] = np.clip(pts[:, :, 0], 0, image_w - 1)
    pts[:, :, 1] = np.clip(pts[:, :, 1], 0, image_h - 1)

    cv2.fillPoly(mask, [pts], color=1)
    return mask


# ═══════════════════════════════════════════════════════════
#  2. AUGMENTATION PIPELINE
# ═══════════════════════════════════════════════════════════

def build_augmentation_pipeline(cfg) -> Optional[A.Compose]:
    """
    Build Albumentations augmentation pipeline for training.

    Applied to BOTH pre and post tiles simultaneously
    (same spatial transforms, independent color transforms).

    Args:
        cfg: Experiment config (augmentation section)

    Returns:
        Albumentations Compose pipeline or None if disabled
    """
    aug_cfg = cfg.augmentation

    if not aug_cfg.enabled:
        return None

    transforms = [
        A.HorizontalFlip(p=aug_cfg.horizontal_flip),
        A.VerticalFlip(p=aug_cfg.vertical_flip),
        A.Rotate(
            limit=aug_cfg.rotation,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.ColorJitter(
            brightness=aug_cfg.brightness,
            contrast=aug_cfg.contrast,
            saturation=aug_cfg.color_jitter,
            hue=0.1,
            p=0.5,
        ),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
    ]

    # For paired images: use same spatial transform on both
    return A.Compose(
        transforms,
        additional_targets={"post_image": "image"},
        bbox_params=A.BboxParams(
            format="pascal_voc",      # [xmin, ymin, xmax, ymax]
            label_fields=["labels"],
            min_visibility=0.1,
        ),
    )


def apply_augmentation(
    pipeline:   A.Compose,
    pre_tile:   np.ndarray,
    post_tile:  np.ndarray,
    boxes:      np.ndarray,
    labels:     np.ndarray,
    masks:      np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply augmentation to a pre/post tile pair with annotations.

    Args:
        pipeline:  Albumentations pipeline
        pre_tile:  HxWx3 uint8
        post_tile: HxWx3 uint8
        boxes:     Nx4 float32 [xmin,ymin,xmax,ymax]
        labels:    N int64
        masks:     NxHxW uint8

    Returns:
        Augmented (pre_tile, post_tile, boxes, labels, masks)
    """
    if pipeline is None or len(boxes) == 0:
        return pre_tile, post_tile, boxes, labels, masks

    # Albumentations expects list of masks
    mask_list = [masks[i] for i in range(len(masks))]

    result = pipeline(
        image      = pre_tile,
        post_image = post_tile,
        bboxes     = boxes.tolist(),
        labels     = labels.tolist(),
        masks      = mask_list,
    )

    aug_pre   = result["image"]
    aug_post  = result["post_image"]
    aug_boxes = np.array(result["bboxes"],  dtype=np.float32) \
                if result["bboxes"] else np.zeros((0, 4), dtype=np.float32)
    aug_labels = np.array(result["labels"], dtype=np.int64) \
                 if result["labels"] else np.zeros(0, dtype=np.int64)
    aug_masks  = np.stack(result["masks"], axis=0) \
                 if result["masks"] else np.zeros(
                     (0, pre_tile.shape[0], pre_tile.shape[1]), dtype=np.uint8
                 )

    return aug_pre, aug_post, aug_boxes, aug_labels, aug_masks


# ═══════════════════════════════════════════════════════════
#  3. TENSOR CONVERSION
# ═══════════════════════════════════════════════════════════

def to_tensor_normalized(image: np.ndarray) -> torch.Tensor:
    """
    Convert HxWx3 uint8 numpy image to normalized CxHxW float tensor.

    Normalization: ImageNet mean/std
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    Args:
        image: HxWx3 uint8 numpy array

    Returns:
        3xHxW float32 tensor
    """
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return torch.from_numpy(image.transpose(2, 0, 1))  # HWC → CHW


def build_target_dict(
    boxes:  np.ndarray,
    masks:  np.ndarray,
    labels: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Build Mask R-CNN compatible target dict from numpy arrays.

    Mask R-CNN expects:
        boxes   : FloatTensor[N, 4]  — [x1, y1, x2, y2]
        labels  : Int64Tensor[N]     — class indices (1-indexed)
        masks   : UInt8Tensor[N,H,W] — binary masks

    Args:
        boxes:  Nx4 float32
        masks:  NxHxW uint8
        labels: N int64 (already 1-indexed from parser)

    Returns:
        Target dict ready for Mask R-CNN
    """
    return {
        "boxes":  torch.as_tensor(boxes,  dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "masks":  torch.as_tensor(masks,  dtype=torch.uint8),
    }


# ═══════════════════════════════════════════════════════════
#  4. DATASET CLASS
# ═══════════════════════════════════════════════════════════

class XBDDataset(Dataset):
    """
    PyTorch Dataset for xBD building damage assessment.

    Returns per __getitem__:
        pre_image  : 3xHxW normalized float tensor (pre-disaster tile)
        post_image : 3xHxW normalized float tensor (post-disaster tile)
        target     : Dict with boxes, labels, masks (Mask R-CNN format)
        tile_info  : TileInfo metadata (for reconstruction)

    Modes:
        train: augmentation ON, cache write
        val:   augmentation OFF, cache read
        test:  augmentation OFF, cache read
    """

    def __init__(
        self,
        cfg,
        split:      str = "train",
        epoch:      int = 0,
    ):
        """
        Args:
            cfg:   Full OmegaConf config
            split: "train" | "val" | "test"
            epoch: Current epoch (controls cache behavior)
        """
        self.cfg       = cfg
        self.split     = split
        self.epoch     = epoch
        self.tile_size = cfg.dataset.tile_size
        self.overlap   = cfg.dataset.overlap
        self.min_area  = cfg.dataset.min_area_ratio
        self.use_cache = cfg.dataset.use_cache

        # Paths
        self.pre_dir    = Path(cfg.paths.pre_dir)
        self.post_dir   = Path(cfg.paths.post_dir)
        self.labels_dir = Path(cfg.paths.labels_dir)

        # Augmentation (train only)
        self.augmentation = (
            build_augmentation_pipeline(cfg)
            if split == "train" else None
        )

        # Tile cache
        config_hash  = get_config_hash(cfg)
        cache_root   = Path(cfg.paths.cache_dir) / split
        self.cache   = TileCache(str(cache_root), config_hash)

        # Build tile index
        self.tile_index = self._build_index()

        console.log(
            f"[green]✓[/green] XBDDataset [{split}] — "
            f"{len(self.tile_index)} tiles from "
            f"{len(self._get_image_ids())} images"
        )

    def _get_image_ids(self) -> List[str]:
        """
        Discover all image IDs from the pre/ directory.

        xBD naming convention:
            pre/  : {disaster}_{tile_id}_pre_disaster.png
            post/ : {disaster}_{tile_id}_post_disaster.png
            labels: {disaster}_{tile_id}_post_disaster.json

        Returns:
            Sorted list of base image IDs
        """
        image_ids = []

        for f in sorted(self.pre_dir.glob("*.png")):
            # Extract base ID: remove "_pre_disaster" suffix
            stem = f.stem
            if "_pre_disaster" in stem:
                base_id = stem.replace("_pre_disaster", "")
                image_ids.append(base_id)

        if len(image_ids) == 0:
            raise RuntimeError(
                f"No images found in {self.pre_dir}. "
                f"Check DATA_DIR in your .env file."
            )

        return image_ids

    def _get_paths(self, image_id: str) -> Tuple[Path, Path, Path]:
        """
        Get pre, post, label paths for a given image ID.

        Returns:
            (pre_path, post_path, label_path)
        """
        pre_path   = self.pre_dir    / f"{image_id}_pre_disaster.png"
        post_path  = self.post_dir   / f"{image_id}_post_disaster.png"
        label_path = self.labels_dir / f"{image_id}_post_disaster.json"

        return pre_path, post_path, label_path

    def _build_index(self) -> List[Tuple[str, int]]:
        """
        Build the tile index: list of (image_id, tile_idx) pairs.

        Strategy (hybrid cache):
            - If cache is ready → load index from cache
            - Else → generate index by scanning all images
              (tiles will be generated and cached in __getitem__)

        Returns:
            List of (image_id, tile_idx) tuples
        """
        if self.use_cache and self.cache.is_ready:
            console.log(
                f"[cyan]→[/cyan] Loading tile index from cache "
                f"({self.split})"
            )
            return self.cache.load_index()

        console.log(
            f"[yellow]→[/yellow] Building tile index for {self.split} "
            f"(will cache on first epoch)"
        )

        image_ids = self._get_image_ids()
        index     = []

        for image_id in image_ids:
            pre_path, post_path, label_path = self._get_paths(image_id)

            if not pre_path.exists() or not post_path.exists():
                warnings.warn(f"Missing image pair for {image_id}, skipping")
                continue

            if not label_path.exists():
                warnings.warn(f"Missing label for {image_id}, skipping")
                continue

            # Load image to get dimensions
            pre_img = np.array(Image.open(pre_path).convert("RGB"))
            h, w    = pre_img.shape[:2]

            # Parse annotations
            ann = parse_xbd_annotation(str(label_path), h, w)

            # Generate tiles for this image (just for index building)
            post_img = np.array(Image.open(post_path).convert("RGB"))

            tiles = image_to_tiles(
                pre_img, post_img,
                ann["boxes"], ann["masks"], ann["labels"],
                image_id       = image_id,
                tile_size      = self.tile_size,
                overlap        = self.overlap,
                min_area_ratio = self.min_area,
                filter_empty   = self.cfg.dataset.filter_empty_tiles,
            )

            for tile in tiles:
                info = tile["tile_info"]
                index.append((info.image_id, info.tile_idx))

                # Cache tiles during index build
                if self.use_cache:
                    self.cache.save_tile(tile)

        # Mark cache as complete
        if self.use_cache and len(index) > 0:
            self.cache.save_index(index)

        return index

    def _load_tile(
        self,
        image_id: str,
        tile_idx: int,
    ) -> Dict[str, Any]:
        """
        Load a tile — from cache if available, else regenerate.

        Args:
            image_id: Source image identifier
            tile_idx: Tile index within the image

        Returns:
            Tile dict with pre_tile, post_tile, boxes, masks, labels, tile_info
        """
        # Try cache first
        if self.use_cache and self.cache.is_ready:
            tile = self.cache.load_tile(image_id, tile_idx)
            if tile is not None:
                return tile

        # Regenerate tile (fallback or first epoch)
        pre_path, post_path, label_path = self._get_paths(image_id)

        pre_img  = np.array(Image.open(pre_path).convert("RGB"))
        post_img = np.array(Image.open(post_path).convert("RGB"))
        h, w     = pre_img.shape[:2]

        ann = parse_xbd_annotation(str(label_path), h, w)

        tiles = image_to_tiles(
            pre_img, post_img,
            ann["boxes"], ann["masks"], ann["labels"],
            image_id       = image_id,
            tile_size      = self.tile_size,
            overlap        = self.overlap,
            min_area_ratio = self.min_area,
            filter_empty   = self.cfg.dataset.filter_empty_tiles,
        )

        # Find the requested tile
        for tile in tiles:
            if tile["tile_info"].tile_idx == tile_idx:
                return tile

        raise RuntimeError(
            f"Tile {tile_idx} not found for image {image_id}"
        )

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch (called by training loop)."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.tile_index)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, TileInfo]:
        """
        Returns:
            pre_tensor  : 3 x tile_size x tile_size float32
            post_tensor : 3 x tile_size x tile_size float32
            target      : {boxes, labels, masks} for Mask R-CNN
            tile_info   : TileInfo for reconstruction
        """
        image_id, tile_idx = self.tile_index[idx]
        tile = self._load_tile(image_id, tile_idx)

        pre_tile   = tile["pre_tile"]
        post_tile  = tile["post_tile"]
        boxes      = tile["boxes"].copy()
        masks      = tile["masks"].copy()
        labels     = tile["labels"].copy()
        tile_info  = tile["tile_info"]

        # Apply augmentation (train split only)
        if self.augmentation is not None and len(boxes) > 0:
            pre_tile, post_tile, boxes, labels, masks = apply_augmentation(
                self.augmentation,
                pre_tile, post_tile,
                boxes, labels, masks,
            )

        # Convert to tensors
        pre_tensor  = to_tensor_normalized(pre_tile)
        post_tensor = to_tensor_normalized(post_tile)
        target      = build_target_dict(boxes, masks, labels)

        return pre_tensor, post_tensor, target, tile_info


# ═══════════════════════════════════════════════════════════
#  5. DATALOADER FACTORY
# ═══════════════════════════════════════════════════════════

def collate_fn(
    batch: List[Tuple],
) -> Tuple[List, List, List, List]:
    """
    Custom collate for Mask R-CNN.

    Mask R-CNN expects a LIST of images (not a batched tensor)
    because each image can have a different number of instances.

    Returns:
        (pre_images, post_images, targets, tile_infos)
        Each is a list of length batch_size
    """
    pre_images  = [item[0] for item in batch]
    post_images = [item[1] for item in batch]
    targets     = [item[2] for item in batch]
    tile_infos  = [item[3] for item in batch]
    return pre_images, post_images, targets, tile_infos


def build_dataloader(
    cfg,
    split:   str = "train",
    epoch:   int = 0,
) -> DataLoader:
    """
    Build a DataLoader for a given split.

    Args:
        cfg:   Full experiment config
        split: "train" | "val" | "test"
        epoch: Current epoch

    Returns:
        Configured PyTorch DataLoader
    """
    dataset = XBDDataset(cfg, split=split, epoch=epoch)

    loader = DataLoader(
        dataset,
        batch_size  = cfg.training.batch_size,
        shuffle     = (split == "train"),
        num_workers = cfg.training.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
        drop_last   = (split == "train"),
    )

    return loader


# ═══════════════════════════════════════════════════════════
#  6. DATASET STATISTICS
# ═══════════════════════════════════════════════════════════

def compute_class_distribution(dataset: XBDDataset) -> Dict[int, int]:
    """
    Count instances per damage class across all tiles.
    Useful for understanding class imbalance.

    Returns:
        Dict mapping class_id → instance count
    """
    counts = {i: 0 for i in range(1, NUM_CLASSES + 1)}

    for idx in range(len(dataset)):
        _, _, target, _ = dataset[idx]
        for label in target["labels"].tolist():
            if label in counts:
                counts[label] += 1

    return counts


# ═══════════════════════════════════════════════════════════
#  7. QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import tempfile
    sys.path.insert(0, str(Path(__file__).parent))

    print("🧪 Testing dataset.py...")

    # Test annotation parser with synthetic data
    import json, tempfile

    # Build a fake xBD annotation file
    fake_ann = {
        "features": {
            "xy": [
                {
                    "properties": {
                        "uid":          "bldg_001",
                        "subtype":      "no-damage",
                        "feature_type": "building"
                    },
                    "wkt": "POLYGON ((100 100, 200 100, 200 200, 100 200, 100 100))"
                },
                {
                    "properties": {
                        "uid":          "bldg_002",
                        "subtype":      "destroyed",
                        "feature_type": "building"
                    },
                    "wkt": "POLYGON ((400 400, 550 400, 550 550, 400 550, 400 400))"
                },
                {
                    "properties": {
                        "uid":          "road_001",
                        "subtype":      "no-damage",
                        "feature_type": "road"       # should be skipped
                    },
                    "wkt": "POLYGON ((0 0, 50 0, 50 50, 0 50, 0 0))"
                }
            ]
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(fake_ann, f)
        ann_path = f.name

    ann = parse_xbd_annotation(ann_path, image_h=1024, image_w=1024)

    assert len(ann["boxes"])  == 2, f"Expected 2 buildings, got {len(ann['boxes'])}"
    assert len(ann["masks"])  == 2
    assert len(ann["labels"]) == 2
    assert ann["labels"][0]   == 1  # no-damage → class 1
    assert ann["labels"][1]   == 4  # destroyed → class 4
    print(f"  ✓ Annotation parser: {len(ann['boxes'])} buildings found")
    print(f"  ✓ Labels: {ann['labels']} (1=no-damage, 4=destroyed)")
    print(f"  ✓ Boxes:  {ann['boxes']}")

    # Test tensor conversion
    img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    t   = to_tensor_normalized(img)
    assert t.shape == (3, 384, 384)
    assert t.dtype == torch.float32
    print(f"  ✓ Tensor conversion: shape={t.shape}, dtype={t.dtype}")

    # Test target dict
    target = build_target_dict(ann["boxes"], ann["masks"], ann["labels"])
    assert "boxes"  in target
    assert "labels" in target
    assert "masks"  in target
    print(f"  ✓ Target dict: boxes={target['boxes'].shape}, "
          f"masks={target['masks'].shape}")

    import os
    os.unlink(ann_path)

    print("✅ dataset.py self-test passed!")
    print()
    print("NOTE: Full dataset test requires xBD data.")
    print("      Set DATA_DIR in .env and run:")
    print("      python src/dataset.py --full-test")