"""
XRayEarth — dataset.py
Seeing through disaster with satellite vision.

Real xBD Dataset Structure:
    xbd/
    ├── tier1/
    │   ├── images/   ← {id}_pre_disaster.png + {id}_post_disaster.png
    │   ├── labels/   ← {id}_pre_disaster.json + {id}_post_disaster.json
    │   └── masks/    ← pre-computed masks (optional)
    ├── tier3/        ← same structure
    ├── hold/         ← same structure (used as val)
    └── test/         ← same structure (used as test)

Split mapping:
    train → tier1/ + tier3/
    val   → hold/
    test  → test/

Key difference from assumption:
    Pre AND post images are in the SAME images/ folder
    Named: {disaster}_{id}_pre_disaster.png
            {disaster}_{id}_post_disaster.png
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

from tiling import TileCache, TileInfo, image_to_tiles
from utils  import get_config_hash, console


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

DAMAGE_LABEL_MAP = {
    "no-damage":     0,
    "minor-damage":  1,
    "major-damage":  2,
    "destroyed":     3,
    "un-classified": 0,
}

CLASS_NAMES = {
    0: "no-damage",
    1: "minor-damage",
    2: "major-damage",
    3: "destroyed",
}

NUM_CLASSES = 4

# xBD split → folder mapping
SPLIT_FOLDERS = {
    "train": ["tier1", "tier3"],
    "val":   ["hold"],
    "test":  ["test"],
}


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

    xBD uses POST disaster labels for damage classification.
    PRE disaster labels only contain building footprints
    (all labeled as 'no-damage').

    Args:
        label_path: Path to *_post_disaster.json
        image_h:    Image height
        image_w:    Image width

    Returns:
        Dict with boxes, masks, labels, uids
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

        if feature_type != "building":
            continue

        uid     = props.get("uid", "unknown")
        subtype = props.get("subtype", "no-damage").lower().strip()

        # Map to 1-indexed class (0 = background in Mask R-CNN)
        damage_class = DAMAGE_LABEL_MAP.get(subtype, 0) + 1

        wkt = feature.get("wkt", "")
        if not wkt:
            continue

        try:
            polygon = parse_wkt_polygon(wkt)
            if polygon is None or polygon.is_empty:
                continue

            coords = np.array(polygon.exterior.coords, dtype=np.float32)
            mask   = rasterize_polygon(coords, image_h, image_w)

            if mask.sum() == 0:
                continue

            ys, xs = np.where(mask)
            xmin, xmax = int(xs.min()), int(xs.max())
            ymin, ymax = int(ys.min()), int(ys.max())

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([float(xmin), float(ymin),
                          float(xmax), float(ymax)])
            masks.append(mask)
            labels.append(damage_class)
            uids.append(uid)

        except Exception as e:
            warnings.warn(f"Skipping {uid}: {e}")
            continue

    if len(boxes) == 0:
        return {
            "boxes":  np.zeros((0, 4), dtype=np.float32),
            "masks":  np.zeros((0, image_h, image_w), dtype=np.uint8),
            "labels": np.zeros(0, dtype=np.int64),
            "uids":   [],
        }

    return {
        "boxes":  np.array(boxes,  dtype=np.float32),
        "masks":  np.stack(masks,  axis=0).astype(np.uint8),
        "labels": np.array(labels, dtype=np.int64),
        "uids":   uids,
    }


def parse_wkt_polygon(wkt: str) -> Optional[Polygon]:
    """Parse WKT polygon string into Shapely Polygon."""
    from shapely import wkt as shapely_wkt
    from shapely.geometry import MultiPolygon

    try:
        geom = shapely_wkt.loads(wkt)
    except Exception:
        return None

    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)

    return geom if isinstance(geom, Polygon) else None


def rasterize_polygon(
    coords:  np.ndarray,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """Rasterize polygon coordinates to binary mask."""
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    pts  = coords[:, :2].astype(np.int32)
    pts  = pts.reshape((-1, 1, 2))
    pts[:, :, 0] = np.clip(pts[:, :, 0], 0, image_w - 1)
    pts[:, :, 1] = np.clip(pts[:, :, 1], 0, image_h - 1)
    cv2.fillPoly(mask, [pts], color=1)
    return mask


# ═══════════════════════════════════════════════════════════
#  2. xBD PATH RESOLVER
# ═══════════════════════════════════════════════════════════

class XBDPathResolver:
    """
    Resolves file paths for the real xBD dataset structure.

    Real structure:
        {data_dir}/{split_folder}/images/{id}_pre_disaster.png
        {data_dir}/{split_folder}/images/{id}_post_disaster.png
        {data_dir}/{split_folder}/labels/{id}_post_disaster.json

    Args:
        data_dir: Root xBD directory (e.g. /data/xbd)
        split:    "train" | "val" | "test"
    """

    def __init__(self, data_dir: str, split: str):
        self.data_dir = Path(data_dir)
        self.split    = split
        self.folders  = SPLIT_FOLDERS[split]

    def get_all_image_ids(self) -> List[Tuple[str, str]]:
        """
        Scan all split folders and return list of
        (folder_name, base_image_id) tuples.

        Returns:
            List of (folder, image_id) e.g.
            [("tier1", "hurricane-harvey_00000001"), ...]
        """
        image_ids = []

        for folder in self.folders:
            img_dir = self.data_dir / folder / "images"

            if not img_dir.exists():
                warnings.warn(f"Folder not found: {img_dir}")
                continue

            # Find all pre-disaster images
            for f in sorted(img_dir.glob("*_pre_disaster.png")):
                base_id = f.stem.replace("_pre_disaster", "")
                image_ids.append((folder, base_id))

        if len(image_ids) == 0:
            raise RuntimeError(
                f"No images found for split '{self.split}' "
                f"in folders {self.folders} under {self.data_dir}.\n"
                f"Check DATA_DIR in your .env file."
            )

        return image_ids

    def get_paths(
        self,
        folder:   str,
        image_id: str,
    ) -> Tuple[Path, Path, Path]:
        """
        Get pre, post, label paths for a given image.

        Args:
            folder:   Split subfolder (e.g. "tier1")
            image_id: Base image ID

        Returns:
            (pre_path, post_path, label_path)
        """
        base     = self.data_dir / folder
        pre_path = base / "images" / f"{image_id}_pre_disaster.png"
        post_path = base / "images" / f"{image_id}_post_disaster.png"
        label_path = base / "labels" / f"{image_id}_post_disaster.json"

        return pre_path, post_path, label_path

    def verify_paths(
        self,
        folder:   str,
        image_id: str,
    ) -> bool:
        """Check all required files exist for an image."""
        pre, post, label = self.get_paths(folder, image_id)
        return pre.exists() and post.exists() and label.exists()


# ═══════════════════════════════════════════════════════════
#  3. AUGMENTATION PIPELINE
# ═══════════════════════════════════════════════════════════

def build_augmentation_pipeline(cfg) -> Optional[A.Compose]:
    """Build Albumentations pipeline for paired pre/post images."""
    if not cfg.augmentation.enabled:
        return None

    aug_cfg = cfg.augmentation
    return A.Compose(
        [
            A.HorizontalFlip(p=aug_cfg.horizontal_flip),
            A.VerticalFlip(p=aug_cfg.vertical_flip),
            A.Rotate(limit=aug_cfg.rotation, p=0.5),
            A.ColorJitter(
                brightness=aug_cfg.brightness,
                contrast=aug_cfg.contrast,
                saturation=aug_cfg.color_jitter,
                p=0.5,
            ),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        ],
        additional_targets={"post_image": "image"},
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.1,
        ),
    )


def apply_augmentation(
    pipeline:  A.Compose,
    pre_tile:  np.ndarray,
    post_tile: np.ndarray,
    boxes:     np.ndarray,
    labels:    np.ndarray,
    masks:     np.ndarray,
) -> Tuple:
    """Apply augmentation to pre/post tile pair."""
    if pipeline is None or len(boxes) == 0:
        return pre_tile, post_tile, boxes, labels, masks

    result = pipeline(
        image      = pre_tile,
        post_image = post_tile,
        bboxes     = boxes.tolist(),
        labels     = labels.tolist(),
        masks      = [masks[i] for i in range(len(masks))],
    )

    aug_boxes = np.array(result["bboxes"],  dtype=np.float32) \
                if result["bboxes"] else np.zeros((0, 4), dtype=np.float32)
    aug_labels = np.array(result["labels"], dtype=np.int64) \
                 if result["labels"] else np.zeros(0, dtype=np.int64)
    aug_masks  = np.stack(result["masks"], axis=0) \
                 if result["masks"] else np.zeros(
                     (0, pre_tile.shape[0], pre_tile.shape[1]), dtype=np.uint8
                 )

    return result["image"], result["post_image"], aug_boxes, aug_labels, aug_masks


# ═══════════════════════════════════════════════════════════
#  4. TENSOR CONVERSION
# ═══════════════════════════════════════════════════════════

def to_tensor_normalized(image: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8 → normalized CxHxW float32 tensor."""
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return torch.from_numpy(image.transpose(2, 0, 1))


def build_target_dict(
    boxes:  np.ndarray,
    masks:  np.ndarray,
    labels: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Build Mask R-CNN target dict from numpy arrays."""
    return {
        "boxes":  torch.as_tensor(boxes,  dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "masks":  torch.as_tensor(masks,  dtype=torch.uint8),
    }


# ═══════════════════════════════════════════════════════════
#  5. DATASET CLASS
# ═══════════════════════════════════════════════════════════

class XBDDataset(Dataset):
    """
    PyTorch Dataset for xBD building damage assessment.

    Handles real xBD folder structure:
        train → tier1/ + tier3/
        val   → hold/
        test  → test/

    Each item returns:
        pre_tensor:  3 x H x W normalized float
        post_tensor: 3 x H x W normalized float
        target:      {boxes, labels, masks}
        tile_info:   TileInfo metadata
    """

    def __init__(self, cfg, split: str = "train", epoch: int = 0):
        self.cfg       = cfg
        self.split     = split
        self.epoch     = epoch
        self.tile_size = cfg.dataset.tile_size
        self.overlap   = cfg.dataset.overlap
        self.min_area  = cfg.dataset.min_area_ratio
        self.use_cache = cfg.dataset.use_cache

        # Path resolver for real xBD structure
        self.resolver = XBDPathResolver(cfg.paths.data_dir, split)

        # Augmentation (train only)
        self.augmentation = (
            build_augmentation_pipeline(cfg)
            if split == "train" else None
        )

        # Tile cache
        config_hash = get_config_hash(cfg)
        cache_root  = Path(cfg.paths.cache_dir) / split
        self.cache  = TileCache(str(cache_root), config_hash)

        # Build tile index
        self.tile_index = self._build_index()

        console.log(
            f"[green]✓[/green] XBDDataset [{split}] — "
            f"{len(self.tile_index)} tiles from "
            f"{len(self.resolver.get_all_image_ids())} images "
            f"({', '.join(SPLIT_FOLDERS[split])})"
        )

    def _build_index(self) -> List[Tuple[str, str, int]]:
        """
        Build tile index: list of (folder, image_id, tile_idx).

        Uses hybrid cache strategy:
            - Cache ready → load from cache
            - Not ready   → generate + cache tiles
        """
        if self.use_cache and self.cache.is_ready:
            console.log(
                f"[cyan]→[/cyan] Loading tile index from cache ({self.split})"
            )
            return self.cache.load_index()

        console.log(
            f"[yellow]→[/yellow] Building tile index for {self.split}..."
        )

        image_ids = self.resolver.get_all_image_ids()
        index     = []

        for folder, image_id in image_ids:
            if not self.resolver.verify_paths(folder, image_id):
                warnings.warn(f"Missing files for {image_id}, skipping")
                continue

            pre_path, post_path, label_path = \
                self.resolver.get_paths(folder, image_id)

            pre_img  = np.array(Image.open(pre_path).convert("RGB"))
            post_img = np.array(Image.open(post_path).convert("RGB"))
            h, w     = pre_img.shape[:2]

            ann = parse_xbd_annotation(str(label_path), h, w)

            tiles = image_to_tiles(
                pre_img, post_img,
                ann["boxes"], ann["masks"], ann["labels"],
                image_id       = f"{folder}/{image_id}",
                tile_size      = self.tile_size,
                overlap        = self.overlap,
                min_area_ratio = self.min_area,
                filter_empty   = self.cfg.dataset.filter_empty_tiles,
            )

            for tile in tiles:
                info = tile["tile_info"]
                index.append((folder, image_id, info.tile_idx))
                if self.use_cache:
                    self.cache.save_tile(tile)

        if self.use_cache and len(index) > 0:
            self.cache.save_index(index)

        return index

    def _load_tile(
        self,
        folder:   str,
        image_id: str,
        tile_idx: int,
    ) -> Dict[str, Any]:
        """Load tile from cache or regenerate."""
        cache_id = f"{folder}/{image_id}"

        if self.use_cache and self.cache.is_ready:
            tile = self.cache.load_tile(cache_id, tile_idx)
            if tile is not None:
                return tile

        # Regenerate
        pre_path, post_path, label_path = \
            self.resolver.get_paths(folder, image_id)

        pre_img  = np.array(Image.open(pre_path).convert("RGB"))
        post_img = np.array(Image.open(post_path).convert("RGB"))
        h, w     = pre_img.shape[:2]

        ann = parse_xbd_annotation(str(label_path), h, w)

        tiles = image_to_tiles(
            pre_img, post_img,
            ann["boxes"], ann["masks"], ann["labels"],
            image_id       = cache_id,
            tile_size      = self.tile_size,
            overlap        = self.overlap,
            min_area_ratio = self.min_area,
            filter_empty   = self.cfg.dataset.filter_empty_tiles,
        )

        for tile in tiles:
            if tile["tile_info"].tile_idx == tile_idx:
                return tile

        raise RuntimeError(
            f"Tile {tile_idx} not found for {image_id}"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.tile_index)

    def __getitem__(self, idx: int):
        folder, image_id, tile_idx = self.tile_index[idx]
        tile = self._load_tile(folder, image_id, tile_idx)

        pre_tile  = tile["pre_tile"].copy()
        post_tile = tile["post_tile"].copy()
        boxes     = tile["boxes"].copy()
        masks     = tile["masks"].copy()
        labels    = tile["labels"].copy()
        tile_info = tile["tile_info"]

        if self.augmentation is not None and len(boxes) > 0:
            pre_tile, post_tile, boxes, labels, masks = apply_augmentation(
                self.augmentation,
                pre_tile, post_tile, boxes, labels, masks,
            )

        pre_tensor  = to_tensor_normalized(pre_tile)
        post_tensor = to_tensor_normalized(post_tile)
        target      = build_target_dict(boxes, masks, labels)

        return pre_tensor, post_tensor, target, tile_info


# ═══════════════════════════════════════════════════════════
#  6. DATALOADER FACTORY
# ═══════════════════════════════════════════════════════════

def collate_fn(batch):
    """Custom collate for Mask R-CNN (returns lists not batches)."""
    return (
        [item[0] for item in batch],
        [item[1] for item in batch],
        [item[2] for item in batch],
        [item[3] for item in batch],
    )


def build_dataloader(cfg, split: str = "train", epoch: int = 0) -> DataLoader:
    """Build DataLoader for a given split."""
    dataset = XBDDataset(cfg, split=split, epoch=epoch)
    return DataLoader(
        dataset,
        batch_size  = cfg.training.batch_size,
        shuffle     = (split == "train"),
        num_workers = cfg.training.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
        drop_last   = (split == "train"),
    )


def compute_class_distribution(dataset: XBDDataset) -> Dict[int, int]:
    """Count instances per damage class across all tiles."""
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
    import sys, json, tempfile
    sys.path.insert(0, str(Path(__file__).parent))

    print("🧪 Testing dataset.py...")

    # Test annotation parser
    fake_ann = {
        "features": {"xy": [
            {
                "properties": {
                    "uid": "bldg_001",
                    "subtype": "no-damage",
                    "feature_type": "building"
                },
                "wkt": "POLYGON ((100 100, 200 100, 200 200, 100 200, 100 100))"
            },
            {
                "properties": {
                    "uid": "bldg_002",
                    "subtype": "destroyed",
                    "feature_type": "building"
                },
                "wkt": "POLYGON ((400 400, 550 400, 550 550, 400 550, 400 400))"
            },
        ]}
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fake_ann, f)
        ann_path = f.name

    ann = parse_xbd_annotation(ann_path, 1024, 1024)
    assert len(ann["boxes"])  == 2
    assert ann["labels"][0]   == 1  # no-damage → 1
    assert ann["labels"][1]   == 4  # destroyed → 4
    print(f"  ✓ Annotation parser: {len(ann['boxes'])} buildings")
    print(f"  ✓ Labels: {ann['labels']} (1=no-damage, 4=destroyed)")

    # Test tensor conversion
    img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    t   = to_tensor_normalized(img)
    assert t.shape == (3, 384, 384)
    print(f"  ✓ Tensor: shape={t.shape}, dtype={t.dtype}")

    # Test path resolver (structure check only)
    print(f"\n  Real xBD split mapping:")
    for split, folders in SPLIT_FOLDERS.items():
        print(f"  ✓ {split:5s} → {folders}")

    import os
    os.unlink(ann_path)

    print("\n✅ dataset.py self-test passed!")
    print("\nNOTE: Full dataset test requires xBD data.")
    print("      Set DATA_DIR=path/to/xbd in .env")
    print("      then run: python src/dataset.py --full-test")
