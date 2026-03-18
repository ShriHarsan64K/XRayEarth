"""
XRayEarth — tiling.py
Seeing through disaster with satellite vision.

Responsibilities:
    - Split high-res satellite images into overlapping tiles
    - Adjust bounding boxes and masks per tile coordinate space
    - Filter empty tiles (no buildings)
    - Hybrid caching: generate on-the-fly epoch 1, cache from epoch 2
    - Reconstruct full-image predictions from tile predictions
    - Handle both pre and post disaster image pairs together
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
from PIL import Image


# ═══════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

class TileInfo:
    """
    Metadata for a single tile cut from a full image.

    Attributes:
        image_id    : Source image identifier
        tile_idx    : Tile index within the image
        x1, y1      : Top-left corner in full image coordinates
        x2, y2      : Bottom-right corner in full image coordinates
        tile_w      : Tile width (pixels)
        tile_h      : Tile height (pixels)
        full_w      : Full image width
        full_h      : Full image height
    """
    def __init__(
        self,
        image_id: str,
        tile_idx: int,
        x1: int, y1: int,
        x2: int, y2: int,
        full_w: int, full_h: int,
    ):
        self.image_id = image_id
        self.tile_idx = tile_idx
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.tile_w = x2 - x1
        self.tile_h = y2 - y1
        self.full_w = full_w
        self.full_h = full_h

    def __repr__(self):
        return (
            f"TileInfo(id={self.image_id}, idx={self.tile_idx}, "
            f"box=({self.x1},{self.y1},{self.x2},{self.y2}))"
        )

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: Dict) -> "TileInfo":
        obj = cls.__new__(cls)
        obj.__dict__.update(d)
        return obj


# ═══════════════════════════════════════════════════════════
#  1. TILE COORDINATE GENERATION
# ═══════════════════════════════════════════════════════════

def generate_tile_coords(
    image_w:   int,
    image_h:   int,
    tile_size: int,
    overlap:   float = 0.15,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate (x1, y1, x2, y2) coordinates for all tiles
    covering a full image with overlap.

    Strategy:
        - Stride = tile_size * (1 - overlap)
        - Last tile in each row/col is snapped to image edge
          (avoids out-of-bounds and ensures full coverage)

    Args:
        image_w:   Full image width
        image_h:   Full image height
        tile_size: Size of each square tile
        overlap:   Fractional overlap between adjacent tiles (0.0–0.5)

    Returns:
        List of (x1, y1, x2, y2) tuples in image coordinates
    """
    stride = int(tile_size * (1.0 - overlap))
    coords = []

    y = 0
    while y < image_h:
        x = 0
        while x < image_w:
            x1 = x
            y1 = y
            x2 = min(x + tile_size, image_w)
            y2 = min(y + tile_size, image_h)

            # Snap small edge tiles back to full tile_size
            # (pad later in extract_tile if needed)
            coords.append((x1, y1, x2, y2))

            if x2 == image_w:
                break
            x += stride

        if y2 == image_h:
            break
        y += stride

    return coords


# ═══════════════════════════════════════════════════════════
#  2. IMAGE TILE EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_tile(
    image:     np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    tile_size: int,
) -> np.ndarray:
    """
    Extract a tile from a numpy image array and pad to tile_size
    if the tile is smaller than tile_size (edge tiles).

    Args:
        image:     HxWxC numpy array
        x1,y1:     Top-left corner
        x2,y2:     Bottom-right corner
        tile_size: Target tile size

    Returns:
        tile_size x tile_size x C numpy array (zero-padded if edge)
    """
    tile = image[y1:y2, x1:x2]

    h, w = tile.shape[:2]
    if h < tile_size or w < tile_size:
        # Zero-pad to tile_size
        c = tile.shape[2] if tile.ndim == 3 else 1
        padded = np.zeros((tile_size, tile_size, c), dtype=tile.dtype)
        padded[:h, :w] = tile
        return padded

    return tile


# ═══════════════════════════════════════════════════════════
#  3. BOUNDING BOX ADJUSTMENT
# ═══════════════════════════════════════════════════════════

def adjust_boxes_to_tile(
    boxes:          np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    min_area_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip bounding boxes to tile boundaries and filter out
    boxes with too little area remaining.

    Args:
        boxes:          Nx4 array of [xmin, ymin, xmax, ymax]
                        in FULL IMAGE coordinates
        x1,y1,x2,y2:   Tile boundaries in full image coordinates
        min_area_ratio: Minimum fraction of original box area
                        that must remain in tile (else discard)

    Returns:
        (adjusted_boxes, valid_mask)
        adjusted_boxes: Nx4 array in TILE coordinates
        valid_mask:     Boolean array of length N
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.array([], dtype=bool)

    tile_w = x2 - x1
    tile_h = y2 - y1

    # Original box areas
    orig_areas = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    # Clip to tile
    clipped         = boxes.copy().astype(np.float32)
    clipped[:, 0]   = np.clip(boxes[:, 0] - x1, 0, tile_w)  # xmin
    clipped[:, 1]   = np.clip(boxes[:, 1] - y1, 0, tile_h)  # ymin
    clipped[:, 2]   = np.clip(boxes[:, 2] - x1, 0, tile_w)  # xmax
    clipped[:, 3]   = np.clip(boxes[:, 3] - y1, 0, tile_h)  # ymax

    # Clipped areas
    clip_areas = (
        (clipped[:, 2] - clipped[:, 0]) *
        (clipped[:, 3] - clipped[:, 1])
    )

    # Keep boxes where enough area remains in tile
    with np.errstate(divide="ignore", invalid="ignore"):
        area_ratio = np.where(orig_areas > 0, clip_areas / orig_areas, 0.0)

    valid_mask = (
        (area_ratio >= min_area_ratio) &
        (clipped[:, 2] > clipped[:, 0]) &
        (clipped[:, 3] > clipped[:, 1])
    )

    return clipped[valid_mask], valid_mask


# ═══════════════════════════════════════════════════════════
#  4. MASK ADJUSTMENT
# ═══════════════════════════════════════════════════════════

def adjust_masks_to_tile(
    masks:     np.ndarray,
    x1: int, y1: int,
    x2: int, y2: int,
    tile_size: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Crop instance masks to tile coordinates and pad to tile_size.

    Args:
        masks:      NxHxW binary mask array (full image size)
        x1,y1,x2,y2: Tile boundaries
        tile_size:  Target output size
        valid_mask: Boolean array from adjust_boxes_to_tile

    Returns:
        N_valid x tile_size x tile_size binary mask array
    """
    if len(masks) == 0 or valid_mask.sum() == 0:
        return np.zeros((0, tile_size, tile_size), dtype=np.uint8)

    valid_masks = masks[valid_mask]
    tile_h      = y2 - y1
    tile_w      = x2 - x1

    result = np.zeros(
        (len(valid_masks), tile_size, tile_size), dtype=np.uint8
    )

    for i, mask in enumerate(valid_masks):
        cropped = mask[y1:y2, x1:x2]
        h, w    = cropped.shape
        result[i, :h, :w] = cropped

    return result


# ═══════════════════════════════════════════════════════════
#  5. TILE FILTERING
# ═══════════════════════════════════════════════════════════

def is_empty_tile(boxes: np.ndarray) -> bool:
    """
    Returns True if a tile has no valid building annotations.

    Args:
        boxes: Adjusted bounding boxes for this tile

    Returns:
        True if tile should be skipped
    """
    return len(boxes) == 0


# ═══════════════════════════════════════════════════════════
#  6. FULL IMAGE → TILES PIPELINE
# ═══════════════════════════════════════════════════════════

def image_to_tiles(
    pre_image:      np.ndarray,
    post_image:     np.ndarray,
    boxes:          np.ndarray,
    masks:          np.ndarray,
    labels:         np.ndarray,
    image_id:       str,
    tile_size:      int,
    overlap:        float,
    min_area_ratio: float,
    filter_empty:   bool = True,
) -> List[Dict[str, Any]]:
    """
    Split a full pre/post image pair into annotated tiles.

    This is the core tiling function called by the dataset.

    Args:
        pre_image:      HxWx3 numpy array (pre-disaster)
        post_image:     HxWx3 numpy array (post-disaster)
        boxes:          Nx4 array [xmin,ymin,xmax,ymax] full coords
        masks:          NxHxW binary masks full image size
        labels:         N array of integer damage labels
        image_id:       Unique identifier for this image pair
        tile_size:      Tile size in pixels
        overlap:        Fractional overlap
        min_area_ratio: Min box area fraction to keep in tile
        filter_empty:   Whether to skip tiles with no buildings

    Returns:
        List of dicts, each containing:
            pre_tile    : tile_size x tile_size x 3 numpy array
            post_tile   : tile_size x tile_size x 3 numpy array
            boxes       : Mx4 float32 array (tile coordinates)
            masks       : M x tile_size x tile_size uint8 array
            labels      : M int64 array
            tile_info   : TileInfo object
    """
    h, w    = pre_image.shape[:2]
    coords  = generate_tile_coords(w, h, tile_size, overlap)
    tiles   = []

    for idx, (x1, y1, x2, y2) in enumerate(coords):

        # Adjust boxes to this tile
        tile_boxes, valid_mask = adjust_boxes_to_tile(
            boxes, x1, y1, x2, y2, min_area_ratio
        )

        # Skip empty tiles if configured
        if filter_empty and is_empty_tile(tile_boxes):
            continue

        # Extract image tiles
        pre_tile  = extract_tile(pre_image,  x1, y1, x2, y2, tile_size)
        post_tile = extract_tile(post_image, x1, y1, x2, y2, tile_size)

        # Adjust masks
        tile_masks = adjust_masks_to_tile(
            masks, x1, y1, x2, y2, tile_size, valid_mask
        )

        # Filter labels
        tile_labels = labels[valid_mask] if len(labels) > 0 else np.array([])

        tile_info = TileInfo(
            image_id = image_id,
            tile_idx = idx,
            x1=x1, y1=y1, x2=x2, y2=y2,
            full_w=w, full_h=h,
        )

        tiles.append({
            "pre_tile":  pre_tile,
            "post_tile": post_tile,
            "boxes":     tile_boxes,
            "masks":     tile_masks,
            "labels":    tile_labels.astype(np.int64),
            "tile_info": tile_info,
        })

    return tiles


# ═══════════════════════════════════════════════════════════
#  7. HYBRID CACHE SYSTEM
# ═══════════════════════════════════════════════════════════

class TileCache:
    """
    Hybrid tile caching system.

    Epoch 1:  Tiles generated on-the-fly, saved to cache dir
    Epoch 2+: Tiles loaded directly from cache (fast I/O)

    Cache invalidation:
        - Config hash stored alongside cache
        - If config changes (tile_size, overlap, etc.),
          cache is automatically regenerated

    Cache structure:
        cache_dir/
            {config_hash}/
                {image_id}_{tile_idx}.pkl
                _cache_meta.json
    """

    def __init__(self, cache_dir: str, config_hash: str):
        self.cache_dir   = Path(cache_dir) / config_hash
        self.config_hash = config_hash
        self._is_ready   = False
        self._meta_path  = self.cache_dir / "_cache_meta.json"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._check_cache()

    def _check_cache(self) -> None:
        """Check if cache exists and is valid."""
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                meta = json.load(f)
            if meta.get("config_hash") == self.config_hash:
                self._is_ready = meta.get("complete", False)

    @property
    def is_ready(self) -> bool:
        """True if cache is fully populated and valid."""
        return self._is_ready

    def _tile_path(self, image_id: str, tile_idx: int) -> Path:
        safe_id = image_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_id}_{tile_idx:04d}.pkl"

    def save_tile(self, tile: Dict[str, Any]) -> None:
        """Save a single tile dict to cache."""
        info = tile["tile_info"]
        path = self._tile_path(info.image_id, info.tile_idx)

        # Serialize TileInfo as dict for pickling
        tile_serializable = {**tile, "tile_info": info.to_dict()}
        with open(path, "wb") as f:
            pickle.dump(tile_serializable, f, protocol=4)

    def load_tile(self, image_id: str, tile_idx: int) -> Optional[Dict]:
        """Load a single tile from cache. Returns None if not found."""
        path = self._tile_path(image_id, tile_idx)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            tile = pickle.load(f)
        # Restore TileInfo object
        tile["tile_info"] = TileInfo.from_dict(tile["tile_info"])
        return tile

    def save_index(self, index: List[Tuple[str, int]]) -> None:
        """
        Save the tile index (list of (image_id, tile_idx) pairs).
        Called once after all tiles are cached.
        """
        index_path = self.cache_dir / "_tile_index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(index, f)

        # Mark cache as complete
        with open(self._meta_path, "w") as f:
            json.dump({
                "config_hash": self.config_hash,
                "complete":    True,
                "tile_count":  len(index),
            }, f, indent=2)

        self._is_ready = True

    def load_index(self) -> List[Tuple[str, int]]:
        """Load tile index from cache."""
        index_path = self.cache_dir / "_tile_index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Tile index not found at {index_path}. "
                f"Cache may be incomplete."
            )
        with open(index_path, "rb") as f:
            return pickle.load(f)

    def clear(self) -> None:
        """Delete all cached tiles (force regeneration)."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._is_ready = False


# ═══════════════════════════════════════════════════════════
#  8. PREDICTION RECONSTRUCTION
# ═══════════════════════════════════════════════════════════

def reconstruct_predictions(
    tile_predictions: List[Dict[str, Any]],
    full_w: int,
    full_h: int,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Merge tile-level predictions back to full image coordinates.

    Strategy:
        1. Translate tile-local box coords → full image coords
        2. Collect all boxes, scores, labels, masks
        3. Apply NMS to remove overlapping duplicates from overlap regions

    Args:
        tile_predictions: List of dicts per tile, each with:
            boxes       : Mx4 tensor (tile coordinates)
            scores      : M tensor
            labels      : M tensor
            masks       : M x tile_h x tile_w tensor
            tile_info   : TileInfo
        full_w, full_h: Full image dimensions
        iou_threshold: NMS IoU threshold

    Returns:
        Dict with boxes, scores, labels, masks in full image coords
    """
    all_boxes  = []
    all_scores = []
    all_labels = []
    all_masks  = []

    for pred in tile_predictions:
        info   = pred["tile_info"]
        boxes  = pred["boxes"]    # Mx4 in tile coords
        scores = pred["scores"]
        labels = pred["labels"]
        masks  = pred["masks"]    # M x tile_h x tile_w

        if len(boxes) == 0:
            continue

        # Translate boxes to full image coordinates
        boxes_full = boxes.clone() if isinstance(boxes, torch.Tensor) \
                     else torch.tensor(boxes, dtype=torch.float32)

        boxes_full[:, 0] += info.x1  # xmin
        boxes_full[:, 1] += info.y1  # ymin
        boxes_full[:, 2] += info.x1  # xmax
        boxes_full[:, 3] += info.y1  # ymax

        # Clamp to full image boundaries
        boxes_full[:, 0].clamp_(0, full_w)
        boxes_full[:, 1].clamp_(0, full_h)
        boxes_full[:, 2].clamp_(0, full_w)
        boxes_full[:, 3].clamp_(0, full_h)

        # Place tile masks into full image canvas
        for i, mask in enumerate(masks):
            full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
            th = min(info.tile_h, mask.shape[0])
            tw = min(info.tile_w, mask.shape[1])
            full_mask[info.y1:info.y1+th, info.x1:info.x1+tw] = \
                mask[:th, :tw]
            all_masks.append(full_mask)

        all_boxes.append(boxes_full)
        all_scores.append(scores)
        all_labels.append(labels)

    if len(all_boxes) == 0:
        return {
            "boxes":  torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.int64),
            "masks":  np.zeros((0, full_h, full_w), dtype=np.uint8),
        }

    # Concatenate all predictions
    all_boxes  = torch.cat(all_boxes,  dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks  = np.stack(all_masks,   axis=0)

    # Apply NMS per class
    from torchvision.ops import nms
    keep_indices = []
    for cls in all_labels.unique():
        cls_mask   = all_labels == cls
        cls_boxes  = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]
        cls_idx    = cls_mask.nonzero(as_tuple=True)[0]

        keep = nms(cls_boxes, cls_scores, iou_threshold)
        keep_indices.append(cls_idx[keep])

    if keep_indices:
        keep = torch.cat(keep_indices)
        keep = keep[all_scores[keep].argsort(descending=True)]
    else:
        keep = torch.tensor([], dtype=torch.long)

    return {
        "boxes":  all_boxes[keep],
        "scores": all_scores[keep],
        "labels": all_labels[keep],
        "masks":  all_masks[keep.numpy()],
    }


# ═══════════════════════════════════════════════════════════
#  9. QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Testing tiling.py...")

    # Synthetic test: 1024×1024 image with 5 buildings
    H, W       = 1024, 1024
    TILE_SIZE  = 384
    OVERLAP    = 0.15

    pre_img  = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    post_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    # 5 synthetic buildings
    boxes = np.array([
        [100, 100, 250, 250],
        [400, 300, 600, 500],
        [700, 100, 900, 300],
        [200, 600, 400, 800],
        [800, 700, 1000, 950],
    ], dtype=np.float32)

    masks  = np.zeros((5, H, W), dtype=np.uint8)
    labels = np.array([0, 1, 2, 3, 0], dtype=np.int64)

    for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        masks[i, y1:y2, x1:x2] = 1

    # Test tile coord generation
    coords = generate_tile_coords(W, H, TILE_SIZE, OVERLAP)
    print(f"  ✓ Generated {len(coords)} tile coordinates")

    # Test full tiling pipeline
    tiles = image_to_tiles(
        pre_img, post_img, boxes, masks, labels,
        image_id       = "test_image_001",
        tile_size      = TILE_SIZE,
        overlap        = OVERLAP,
        min_area_ratio = 0.1,
        filter_empty   = True,
    )
    print(f"  ✓ Generated {len(tiles)} non-empty tiles")
    print(f"  ✓ First tile pre shape:  {tiles[0]['pre_tile'].shape}")
    print(f"  ✓ First tile post shape: {tiles[0]['post_tile'].shape}")
    print(f"  ✓ First tile boxes:      {tiles[0]['boxes'].shape}")
    print(f"  ✓ First tile masks:      {tiles[0]['masks'].shape}")
    print(f"  ✓ First tile info:       {tiles[0]['tile_info']}")

    # Test cache
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TileCache(tmpdir, config_hash="abc12345")
        assert not cache.is_ready

        index = []
        for tile in tiles:
            cache.save_tile(tile)
            info = tile["tile_info"]
            index.append((info.image_id, info.tile_idx))

        cache.save_index(index)
        assert cache.is_ready
        print(f"  ✓ Cache saved: {len(index)} tiles")

        loaded = cache.load_tile("test_image_001", tiles[0]["tile_info"].tile_idx)
        assert loaded is not None
        print(f"  ✓ Cache load successful")

    print("✅ tiling.py self-test passed!")