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
from PIL import Image
from src.helpers.wall_detection_inf import WallDetection
from pathlib import Path

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
    mask = mask.resize(pil_img.size, Image.Resampling.BILINEAR)
    mask = np.array(mask)
    return mask.astype(np.float32)


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

    detector = WallDetection(ckpt_path=args.ckpt_path)
    detector.pipe.to(device)

    windows = list(sliding_windows(W, H, tile_px, stride_px))
    for (x0, y0, x1, y1) in tqdm(windows, desc="Tiled inference"):
        tile_img = im_full.crop((x0, y0, x1, y1))
        m = infer_tile(detector, tile_img, args.num_images)
        tile_vis = np.zeros_like(acc)
        tile_vis[y0:y1, x0:x1] = m[: y1 - y0, : x1 - x0]
        tile_vis = 255 - tile_vis
        tile_vis = Image.fromarray(tile_vis.astype(np.uint8), mode="L")
        tile_vis.save(f"{args.output[:-4]}_tile_{x0}_{y0}.png")
        acc[y0:y1, x0:x1] += 255 - m[: y1 - y0, : x1 - x0]
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
