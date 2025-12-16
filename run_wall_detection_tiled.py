#!/usr/bin/env python3
"""
Tiled WAFFLE wall-segmentation with constrained tile grid sizing.

 • --mpp <float> gives metres-per-pixel for the input floor plan.
 • --max_tiles_x/y <int> limit the tile grid (default 3x3, recommended 2x2 to 4x4).
   The script calculates the optimal tile size to cover the floorplan with at most
   this many tiles, making tiles as large as possible while staying within bounds.
 • --min_tile_m <float> optionally sets a minimum tile size in metres.
   If not specified, it's auto-calculated from max_tiles to fit the floorplan.
 • --max_sliding_tiles <int> maximum total sliding windows to process (default 12).
   If overlap would create more windows, overlap is automatically reduced to stay within limit.
 
Note: Performance degrades with very large tiles. Keep grid to 2x2-4x4 for best results.
"""
import argparse, math, os, tempfile
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image, ImageFilter
from src.helpers.wall_detection_inf import WallDetection
from pathlib import Path
from skimage.morphology import skeletonize
import cv2

def crop_to_content(img: np.ndarray,
                    bg_threshold=1,
                    padding=0):
    """
    GIMP-style “Crop to content” with optional padding.

    Parameters
    ----------
    img          : BGR[A] image loaded with cv2.imread(...)
    bg_threshold : int/float · 0–255 (or 0–1 for float images)
    padding      : int  or  (top, right, bottom, left)

    Returns
    -------
    cropped : np.ndarray
    bbox    : tuple[int, int, int, int]   # (x, y, w, h) in original coords
    """

    h, w = img.shape[:2]

    # ---------- Build foreground mask ----------
    if img.shape[2] == 4:                       # 1) via alpha if present
        mask = img[:, :, 3] > bg_threshold
    else:                                       # 2) via colour diff vs corner
        bg_color = img[0:4, 0:4].reshape(-1, 3).mean(0)  # avg of 4×4 px corner
        diff = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        bg_luma = int(np.dot(bg_color, [0.114, 0.587, 0.299]))
        mask = np.abs(diff - bg_luma) > bg_threshold

    if not mask.any():                          # empty → nothing to crop
        return img.copy(), (0, 0, w, h)

    # ---------- Tight bounding rectangle ----------
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    # ---------- Handle padding ----------
    if isinstance(padding, int):
        top = right = bottom = left = padding
    else:                                       # assume len==4
        top, right, bottom, left = padding

    y0 = max(0, y0 - top)
    x0 = max(0, x0 - left)
    y1 = min(h, y1 + bottom)
    x1 = min(w, x1 + right)

    cropped = img[y0:y1, x0:x1].copy()
    return cropped, (x0, y0, x1 - x0, y1 - y0)

# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser(
        description="Run WAFFLE wall segmentation on a floor-plan using real-world-aware tiled inference"
    )
    p.add_argument("--ckpt_path", type=str,
                   default="checkpoints/checkpoint-200000/controlnet",
                   help="Path to the extracted ControlNet folder")
    p.add_argument("--input", "-i", type=str, required=True,
                   help="Path to the cluttered floor-plan PNG")
    p.add_argument("--output", "-o", type=str, required=True,
                   help="Where to save the walls-only PNG")
    p.add_argument("--num_images", "-n", type=int, default=1,
                   help="How many samples to draw per tile (averaged)")
    p.add_argument("--mpp", type=float, required=True,
                   help="Metres per pixel of the input plan (e.g. 0.01 for 1 cm / px)")
    p.add_argument("--min_tile_m", type=float, default=None,
                   help="Minimum tile size in metres (default: auto-calculated based on max_tiles)")
    p.add_argument("--max_tiles_x", type=int, default=3,
                   help="Maximum number of tiles in X direction (default: 3)")
    p.add_argument("--max_tiles_y", type=int, default=3,
                   help="Maximum number of tiles in Y direction (default: 3)")
    p.add_argument("--overlap", type=float, default=0.2,
                   help="Tile overlap fraction between 0 and 0.5 (e.g. 0.25 = 25 %)")
    p.add_argument("--max_sliding_tiles", type=int, default=12,
                   help="Maximum total number of sliding windows to process (default: 12)")
    return p.parse_args()


# ---------- helpers ----------
def sliding_windows(w_px, h_px, tile_px, stride_px):
    """Yield (x0, y0, x1, y1) boxes covering an image of size (w_px, h_px)."""
    nx = math.ceil((w_px - tile_px) / stride_px) + 1
    ny = math.ceil((h_px - tile_px) / stride_px) + 1
    nx, ny = max(1, nx), max(1, ny)  # at least one tile in each direction
    print (f"Sliding windows: {nx} x {ny} = {nx * ny} tiles")
    for iy in range(ny):
        for ix in range(nx):
            x0 = int(ix * stride_px)
            y0 = int(iy * stride_px)
            x1 = min(x0 + tile_px, w_px)
            y1 = min(y0 + tile_px, h_px)
            if x1 - x0 < tile_px:  # shift window if clipped at right edge
                x0 = max(0, x1 - tile_px)
            if y1 - y0 < tile_px:  # shift window if clipped at bottom edge
                y0 = max(0, y1 - tile_px)
            yield x0, y0, x1, y1


def infer_tile(detector, pil_img, num_images):
    """Run ControlNet inference on a PIL image tile and return a float32 mask array."""
    mask = detector.infer_pil(image=pil_img, num_images=num_images)
    import matplotlib.pyplot as plt
    # plt.imshow(np.asarray(mask))
    # plt.show()
    mask_dilated = mask.filter(ImageFilter.MaxFilter(size=3))  # dilate to fill gaps
    # plt.imshow(np.asarray(mask_dilated))
    # plt.show()
    mask_eroded = mask_dilated.filter(ImageFilter.MinFilter(size=3))  # erode to remove noise
    mask_eroded_max = mask_eroded.filter(ImageFilter.MinFilter(size=5)).filter(ImageFilter.MinFilter(size=5)).filter(ImageFilter.MinFilter(size=5))
    # plt.imshow(np.asarray(mask_eroded))
    # plt.show()
    # mask = mask_eroded.resize(pil_img.size, Image.Resampling.BILINEAR)
    mask = np.array(mask_eroded)
    mask = 255 - mask  # invert: white walls (255) on black bg (0)
    skel = skeletonize((255-np.array(mask_eroded_max)) > 0)
    dist_img = cv2.distanceTransform(1-(skel > 0).astype(np.uint8), cv2.DIST_L2, 5, dstType=cv2.CV_32F)

    # mask_dist = dist_img[mask > 0]
    mean_mask_dist = dist_img[mask > 0].sum() / skel.sum()  if skel.sum() > 0 else 0.0
    mask = cv2.resize(mask, pil_img.size, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("Test", mask.astype(np.uint8) * 255)
    # cv2.waitKey(0)
    # plt.imshow(dist_img)
    # plt.show()
    # plt.imshow(mask_eroded_max)
    # plt.show()
    if mean_mask_dist < 60:
        return mask.astype(np.float32), mean_mask_dist  # return as is if it looks like a wall
    else:
        return np.zeros_like(mask, dtype=np.float32), 0  # return empty mask if not wall-like


# ---------- main ----------
def get_wall_mask(input: Path, output: Path, mpp: float, min_tile_m, max_tiles_x, max_tiles_y, 
                  overlap, num_images, ckpt_path, max_sliding_tiles):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    im_full = Image.open(input).convert("RGB")
    W, H = im_full.size
    
    trimmed, box = crop_to_content(np.asarray(im_full), padding=int(0.1 * min(H, W)))
    
    detector = WallDetection(ckpt_path=ckpt_path)
    detector.pipe.to(device)

    w, h = box[2], box[3]
    bx, by = box[0], box[1]
    
    # Calculate actual floorplan dimensions in meters
    floorplan_width_m = w * mpp
    floorplan_height_m = h * mpp
    
    print(f"Floorplan dimensions: {floorplan_width_m:.1f}m x {floorplan_height_m:.1f}m")
    print(f"Max tile grid: {max_tiles_x}x{max_tiles_y}")
    
    # Calculate optimal tile grid
    # Goal: Use minimum number of tiles (starting from 2x2) while respecting min_tile_m constraint
    
    # If min_tile_m not specified, calculate based on max_tiles
    if min_tile_m is None:
        # Start with the constraint that we want at most max_tiles_x × max_tiles_y
        # Calculate minimum tile size needed
        min_tile_m_x = floorplan_width_m / max_tiles_x
        min_tile_m_y = floorplan_height_m / max_tiles_y
        min_tile_m = max(min_tile_m_x, min_tile_m_y)
        print(f"Auto-calculated min_tile_m: {min_tile_m:.2f}m (to fit {max_tiles_x}x{max_tiles_y} grid)")
    else:
        print(f"Using specified min_tile_m: {min_tile_m:.2f}m")
    
    # Convert to pixels
    min_tile_px = max(64, int(round(min_tile_m / mpp)))  # enforce minimum 64px
    
    # Calculate how many tiles we'd need with this minimum tile size
    tiles_x_needed = max(1, math.ceil(floorplan_width_m / min_tile_m))
    tiles_y_needed = max(1, math.ceil(floorplan_height_m / min_tile_m))
    
    # Clamp to max_tiles constraints
    tiles_x = min(tiles_x_needed, max_tiles_x)
    tiles_y = min(tiles_y_needed, max_tiles_y)
    
    # Ensure at least 2x2 for better stitching (unless floorplan is tiny)
    if tiles_x == 1 and floorplan_width_m > min_tile_m * 1.5:
        tiles_x = 2
    if tiles_y == 1 and floorplan_height_m > min_tile_m * 1.5:
        tiles_y = 2
    
    # Calculate actual tile size in pixels based on grid
    tile_width_px = max(min_tile_px, w // tiles_x)
    tile_height_px = max(min_tile_px, h // tiles_y)
    tile_px = min(tile_width_px, tile_height_px)  # use smaller dimension for square tiles
    
    actual_tile_m = tile_px * mpp
    
    print(f"Final tile grid: {tiles_x}x{tiles_y} ({tiles_x * tiles_y} tiles total)")
    print(f"Tile size: {tile_px}px ≈ {actual_tile_m:.2f}m")
    
    # Calculate stride and adjust overlap if necessary to respect max_sliding_tiles
    stride_px = max(1, int(tile_px * (1.0 - overlap)))
    
    # Calculate how many windows this would generate
    nx = math.ceil((w - tile_px) / stride_px) + 1
    ny = math.ceil((h - tile_px) / stride_px) + 1
    nx, ny = max(1, nx), max(1, ny)
    total_windows = nx * ny
    
    actual_overlap = overlap
    if total_windows > max_sliding_tiles:
        print(f"Initial overlap {overlap*100:.0f}% would create {total_windows} windows (exceeds max {max_sliding_tiles})")
        
        # Search from requested stride (max overlap) to tile_px (no overlap)
        min_stride = stride_px  # Start from requested overlap
        max_stride = tile_px     # No overlap (stride = tile size)
        
        best_stride = max_stride  # Default to no overlap if nothing works
        for test_stride in range(min_stride, max_stride + 1):
            test_nx = math.ceil((w - tile_px) / test_stride) + 1
            test_ny = math.ceil((h - tile_px) / test_stride) + 1
            test_nx, test_ny = max(1, test_nx), max(1, test_ny)
            test_total = test_nx * test_ny
            
            if test_total <= max_sliding_tiles:
                best_stride = test_stride
                nx, ny = test_nx, test_ny
                total_windows = test_total
                break
        
        stride_px = best_stride
        actual_overlap = 1.0 - (stride_px / tile_px)
        
        if actual_overlap <= 0.001:  # essentially no overlap
            print(f"⚠ Adjusted to NO OVERLAP (0%) → {total_windows} windows ({nx}x{ny})")
        else:
            print(f"Adjusted overlap to {actual_overlap*100:.1f}% → {total_windows} windows ({nx}x{ny})")
    else:
        print(f"Using {total_windows} windows ({nx}x{ny}) with {overlap*100:.0f}% overlap")
    
    print(f"Stride: {stride_px}px")
    
    # Use tiled processing
    acc = np.zeros((H, W), dtype=np.float32)
    wmap = np.zeros((H, W), dtype=np.float32)

    windows = list(sliding_windows(w, h, tile_px, stride_px))
    
    for (x0, y0, x1, y1) in tqdm(windows, desc="Tiled inference"):
        x0, y0, x1, y1 = bx + x0, by + y0, bx + x1, by + y1
        tile_img = im_full.crop((x0, y0, x1, y1))
        m, mean_mask_dist = infer_tile(detector, tile_img, num_images)
        tile_vis = np.zeros_like(acc)
        tile_vis[y0:y1, x0:x1] = m[: y1 - y0, : x1 - x0]
        # tile_vis = Image.fromarray(tile_vis.astype(np.uint8), mode="L")                                                  
        # tile_vis.save(f"{output[:-4]}_tile_{x0}_{y0}.png")
        # print (f"Tile ({x0}, {y0}) mean_mask_dist: {mean_mask_dist:.4f}")
        acc[y0:y1, x0:x1] += m[: y1 - y0, : x1 - x0]
        wmap[y0:y1, x0:x1] += 1.0

    # Properly average overlapping regions - each pixel gets the mean of all tiles that covered it
    averaged = np.where(wmap > 0, acc / wmap, 0)
    merged = (averaged > 127).astype(np.uint8) * 255
    Image.fromarray(merged, mode="L").save(output)

    print(f"✔ Saved stitched wall mask to {output}")
    print(f"   • mpp:      {mpp:.6f} m/px")
    print(f"   • tile:     {tile_px}px  ≈ {tile_px*mpp:.2f} m")
    print(f"   • overlap:  {actual_overlap*100:.1f} %")


if __name__ == "__main__":
    args = get_args()
    min_tile_m = args.min_tile_m 
    get_wall_mask(Path(args.input), Path(args.output), args.mpp, min_tile_m, 
                  args.max_tiles_x, args.max_tiles_y, args.overlap, args.num_images, 
                  args.ckpt_path, args.max_sliding_tiles)
