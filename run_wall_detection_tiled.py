#!/usr/bin/env python3
"""
Tiled WAFFLE wall-segmentation with real-world-aware tile sizing.

 • --mpp <float> gives metres-per-pixel for the input floor plan.
 • --tile_m <float> sets the real-world edge length of each tile (default 3 m).
   The script converts this to pixels:  tile_px = round(tile_m / mpp).
 • All other behaviour (overlap %, inversion convention, etc.) is unchanged.
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
    p.add_argument("--input", "-i", required=True,
                   help="Path to the cluttered floor-plan PNG")
    p.add_argument("--output", "-o", required=True,
                   help="Where to save the walls-only PNG")
    p.add_argument("--num_images", "-n", type=int, default=1,
                   help="How many samples to draw per tile (averaged)")
    p.add_argument("--mpp", type=float, required=True,
                   help="Metres per pixel of the input plan (e.g. 0.01 for 1 cm / px)")
    p.add_argument("--tile_m", type=float, default=10.0,
                   help="Edge length of a square tile in metres (default 3 m)")
    p.add_argument("--overlap", type=float, default=0.2,
                   help="Tile overlap fraction between 0 and 0.5 (e.g. 0.25 = 25 %)")
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
    # 3. “Line-likeness” of the mask
    skel = skeletonize((255-np.array(mask_eroded_max)) > 0)
    # dist_img = cv2.distanceTransform(1 - skel.astype(np.uint8), cv2.DIST_L2, 5, dstType=cv2.CV_32F)
    dist_img = cv2.distanceTransform(1-(skel > 0).astype(np.uint8), cv2.DIST_L2, 5, dstType=cv2.CV_32F)
    # 4. Chamfer score of mask in distance transform space
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
def main():
    args = get_args()

    # Convert physical tile size (m) to pixels
    tile_px = int(round(args.tile_m / args.mpp))
    if tile_px <= 0:
        raise ValueError("tile_m divided by mpp produced a non-positive pixel size")
    stride_px = int(tile_px * (1.0 - args.overlap))
    stride_px = max(1, stride_px)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    im_full = Image.open(args.input).convert("RGB")
    W, H = im_full.size
    acc = np.zeros((H, W), dtype=np.float32)
    wmap = np.zeros((H, W), dtype=np.float32)

    trimmed, box = crop_to_content(np.asarray(im_full), padding=int(0.1 * min(H, W)))

    detector = WallDetection(ckpt_path=args.ckpt_path)
    detector.pipe.to(device)

    w, h = box[2], box[3]
    bx, by = box[0], box[1]

    print (f"Running tiled inference on {W}x{H} image with {tile_px}px tiles and {args.overlap*100:.0f}% overlap... with stride {stride_px}px")

    windows = list(sliding_windows(w, h, tile_px, stride_px))
    for (x0, y0, x1, y1) in tqdm(windows, desc="Tiled inference"):
        x0, y0, x1, y1 = bx + x0, by + y0, bx + x1, by + y1
        tile_img = im_full.crop((x0, y0, x1, y1))
        m, mean_mask_dist = infer_tile(detector, tile_img, args.num_images)
        tile_vis = np.zeros_like(acc)
        tile_vis[y0:y1, x0:x1] = m[: y1 - y0, : x1 - x0]
        tile_vis = Image.fromarray(tile_vis.astype(np.uint8), mode="L")
        tile_vis.save(f"{args.output[:-4]}_tile_{x0}_{y0}.png")
        print (f"Tile ({x0}, {y0}) mean_mask_dist: {mean_mask_dist:.4f}")
        acc[y0:y1, x0:x1] += m[: y1 - y0, : x1 - x0]
        wmap[y0:y1, x0:x1] += 1.0

    # merged = np.where(wmap > 0, acc / wmap, 0).astype(np.uint8)
    merged = (acc > 127).astype(np.uint8) * 255  # binary mask
    Image.fromarray(merged, mode="L").save(args.output)

    print(f"✔ Saved stitched wall mask to {args.output}")
    print(f"   • mpp:      {args.mpp:.6f} m/px")
    print(f"   • tile:     {tile_px}px  ≈ {tile_px*args.mpp:.2f} m")
    print(f"   • overlap:  {args.overlap*100:.0f} %")


if __name__ == "__main__":
    main()
