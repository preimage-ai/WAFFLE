#!/usr/bin/env python3
"""
realtime_chamfer_align_scale.py
--------------------------------
Chamfer-matching registration with live visualisation *and* scale search.

Keys:
  q / Esc   – abort early and keep best pose so far
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ──────────────── helpers ────────────────
def load_binary(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    if img is None:
        raise FileNotFoundError(path)
    _, bw = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw.astype(np.uint8)


def edge_map(mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(mask * 255, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return (edges > 0).astype(np.uint8)

def distance_transform(edge: np.ndarray) -> np.ndarray:
    inv = 1 - edge
    return cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)


def chamfer_score(dist_map: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    # import matplotlib.pyplot as plt
    # plt.imshow(tmpl, cmap="gray")
    # plt.title("Chamfer Template")
    # plt.show()
    # plt.imshow(dist_map, cmap="gray")
    # plt.title("Distance Map")
    # plt.show()

    return cv2.filter2D(dist_map, cv2.CV_32F, tmpl, borderType=cv2.BORDER_REPLICATE)


def render_pointcloud(cloud_path: str,
                      mpp: float,
                      background=(0, 0, 0, 0),
                      color_by="height",
                      out_path: str | None = None):
    """
    Orthographic bird’s-eye render of a point cloud using Open3D.

    Parameters
    ----------
    cloud_path : str
        File path to the point cloud (any format Open3D can read).
    mpp : float
        Metres per pixel in the XY plane (orthographic scale).
    background : 3- or 4-tuple[int]  (default black, transparent)
        RGBA background colour.
    color_by : {"height", "intensity", "rgb"}
        How to colour points in the render.
    out_path : str | None
        If given, the rendered PNG will be saved here.

    Returns
    -------
    img_np : np.ndarray  (H, W, 4)  uint8
        The rendered RGBA image.
    cam_dict : dict
        Camera intrinsics+extrinsics used for reproducibility.
    """

    # ------------------------------------------------------------
    # 1. Load point cloud and basic stats
    # ------------------------------------------------------------
    pcd = o3d.t.io.read_point_cloud(cloud_path)          # Open3D I/O  :contentReference[oaicite:0]{index=0}
    seg = pcd.point.seg.numpy().reshape(-1)
    normals = pcd.point.normals.numpy()
    
    mask = np.ones_like(seg).astype(np.bool_)
    mask = np.logical_and(mask, np.logical_or(seg == 0, seg == 1, seg == 32))
    mask = np.logical_and(mask, np.abs(normals[:, 2]) < 0.3)
    
    pcd = o3d.io.read_point_cloud(cloud_path)
    # pcd.rotate(np.asarray(R.from_euler('x', 90, degrees=True).as_matrix()))
    pts = np.asarray(pcd.points)[mask]
    pcd.points = o3d.utility.Vector3dVector(pts)
    if pts.size == 0:
        raise ValueError("Point cloud is empty!")

    # XY extent → pixel resolution
    min_xy = pts[:, :2].min(0)
    max_xy = pts[:, :2].max(0)
    extent_xy = max_xy - min_xy
    width_px = int(np.ceil(extent_xy[0] / mpp))
    height_px = int(np.ceil(extent_xy[1] / mpp))

    # ------------------------------------------------------------
    # 2. Prepare colours (height, intensity or original RGB)
    # ------------------------------------------------------------
    # if color_by == "height":
    #     z = pts[:, 2]
    #     z_norm = (z - z.min()) / (z.ptp() + 1e-9)
    #     colours = np.stack([z_norm, 1-z_norm, 0.5*np.ones_like(z_norm)], 1)
    # elif color_by == "intensity" and pcd.has_intensities():
    #     inten = np.asarray(pcd.intensities)
    #     inten = (inten - inten.min()) / (inten.ptp() + 1e-9)
    #     colours = np.stack([inten]*3, 1)
    # elif pcd.has_colors():
    #     colours = np.asarray(pcd.colors)
    # else:
    colours = np.ones_like(pts)    # light grey fallback
    pcd.colors = o3d.utility.Vector3dVector(colours)

    # ------------------------------------------------------------
    # 3. Build an off-screen scene and orthographic camera
    # ------------------------------------------------------------
    # 3.1 Renderer & scene
    renderer = o3d.visualization.rendering.OffscreenRenderer(width_px,
                                                             height_px) 
    scene = renderer.scene
    scene.set_background(background)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"                        # colour exactly as set
    mat.point_size = 1.0
    scene.add_geometry("cloud", pcd, mat)

    # 3.2 Camera: look straight down –Z  (world Z=up convention)
    bbox   = pcd.get_axis_aligned_bounding_box()
    centre = bbox.get_center()
    eye    = centre + np.array([0, 0, bbox.get_extent()[2] * 2])
    up     = np.array([0, 1, 0])

    cam = scene.camera
    # ortho frustum boundaries (left, right, bottom, top, near, far)
    cam.set_projection(o3d.visualization.rendering.Camera.Projection.Ortho,
                       -extent_xy[0]/2, extent_xy[0]/2,
                       -extent_xy[1]/2, extent_xy[1]/2,
                       0.1, bbox.get_extent()[2]*4)   
    cam.look_at(centre, eye, up)                  

    # ------------------------------------------------------------
    # 4. Render
    # ------------------------------------------------------------
    img_o3d = renderer.render_to_image()               
    img_np = np.asarray(img_o3d)                       # uint8 RGBA

    # optional disk write
    if out_path:
        o3d.io.write_image(out_path, img_o3d)

    # clamp renderer resources
    # renderer.close()

    cam_dict = dict(width=width_px, height=height_px,
                    left=-extent_xy[0]/2, right=extent_xy[0]/2,
                    bottom=-extent_xy[1]/2, top=extent_xy[1]/2,
                    near=0.1, far=bbox.get_extent()[2]*4,
                    eye=eye.tolist(), centre=centre.tolist(), up=up.tolist())
    
    _, bw = cv2.threshold(img_np.mean(axis=-1).astype(np.uint8), 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    import matplotlib.pyplot as plt
    plt.imshow(bw)
    plt.show()
    return bw, cam_dict


def downscale_img(img, mpp, side=512):
    tgtsize = side * side
    imgsize = img.shape[0] * img.shape[1]
    fac = np.sqrt(tgtsize / imgsize)
    img_resized = cv2.resize(img, (int(img.shape[1] * fac), int(img.shape[0] * fac)), interpolation=cv2.INTER_LINEAR)
    mpp_new = mpp / fac
    return img_resized, mpp_new

# ──────────────── main ────────────────
def align_cloud(fixed: Path, args_mpp: float, cloud: Path, rot_range = (-180, 180), rot_step = 1.0, scale_range = (0.8, 1.4), scale_step = 0.02):
    mpp = 0.05
    moving_bw, cam_info = render_pointcloud(cloud, mpp, out_path="test.png")
    # import matplotlib.pyplot as plt
    # plt.imshow(moving_bw)
    # plt.show()

    fixed_bw = load_binary(fixed)
    fixed_bw, mpp_fixed = downscale_img(fixed_bw, mpp=args_mpp)
    # moving_bw = load_binary(args.moving)

    fac = mpp / mpp_fixed
    # fac = 1
    print ("mpp_fixed: ", mpp_fixed)
    # breakpoint()
    moving_bw = cv2.resize(moving_bw, (int(moving_bw.shape[1] * fac), int(moving_bw.shape[0] * fac)), interpolation=cv2.INTER_LINEAR)
    print ("mpp", mpp)
    cv2.namedWindow("tst", cv2.WINDOW_NORMAL)
    cv2.imshow("tst", (moving_bw * 255).astype(np.uint8))
    cv2.waitKey(0)

    # fixed_edges = edge_map(fixed_bw)
    # moving_edges = edge_map(moving_bw)

    recall_dist = 1 / (1+distance_transform(fixed_bw))
    dist_min = 0
    recall_dist[:, 0] = dist_min
    recall_dist[:, -1] = dist_min
    recall_dist[0, :] = dist_min
    recall_dist[-1, :] = dist_min

    accuracy_dist = distance_transform(fixed_bw)
    dist_max = accuracy_dist.max()
    accuracy_dist[:, 0] = dist_max
    accuracy_dist[:, -1] = dist_max
    accuracy_dist[0, :] = dist_max
    accuracy_dist[-1, :] = dist_max

    h, w = fixed_bw.shape
    cx, cy = w / 2.0, h / 2.0

    cv2.namedWindow("Chamfer Cost", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Overlay Preview", cv2.WINDOW_NORMAL)

    best = {"score": -np.inf, "angle": 0.0, "scale": 1.0, "dx": 0, "dy": 0}

    rot_angles = np.arange(rot_range[0], rot_range[1] + 1e-6, rot_step)
    scales = np.arange(scale_range[0], scale_range[1] + 1e-6, scale_step)

    chamfers = {}
    for s in tqdm(scales, desc="scale loop"):
        chamfers[s] = {}
        for ang in rot_angles:
            # single warp that includes *both* rotation and scale
            M_rs = cv2.getRotationMatrix2D((cx, cy), ang, s)
            rotated = cv2.warpAffine(moving_bw, M_rs, (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # translation search via Chamfer convolution
            recall_score_map = chamfer_score(recall_dist, rotated)
            accuracy_score_map = chamfer_score(accuracy_dist, rotated)
            score_map = recall_score_map / (accuracy_score_map + 1e-9) ** (1/3)
            score_map[accuracy_score_map < 1] = 0  # in case accuracy score map randomly give a very small value
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(score_map)

            if max_val >  best["score"]:
                best.update(score=max_val, angle=ang, scale=s,
                            dx=max_loc[0] - cx, dy=max_loc[1] - cy)
            chamfers[s][ang] = max_val
            # ─── live visualisation ───
            cost_vis = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX)
            cost_vis = cv2.applyColorMap(cost_vis.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.circle(cost_vis, max_loc, 4, (0, 0, 255), -1)
            cv2.putText(cost_vis, f"rot {ang:.1f}  s {s:.3f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow("Chamfer Cost", cost_vis)

            # if ang > -84.5 and ang < 84.5 and s > 1.35 and s < 1.45:
            #     breakpoint()
            #     cv2.waitKey(0)  # pause for inspection

            M_tmp = M_rs.copy()
            M_tmp[0, 2] += max_loc[0] - cx
            M_tmp[1, 2] += max_loc[1] - cy
            warped_edge = cv2.warpAffine(moving_bw * 255, M_tmp, (w, h),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            dist_uint = (recall_dist * 255 / recall_dist.max()).astype(np.uint8)
            overlay = cv2.applyColorMap(dist_uint, cv2.COLORMAP_TURBO)
            overlay[fixed_bw > 0] = (0, 0, 0) 
            # cnt, _ = cv2.findContours(warped_edge, cv2.RETR_EXTERNAL,
            #                           cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(overlay, cnt, -1, (0, 255, 0), 1)
            overlay[warped_edge > 0] = (0, 0, 255)
            cv2.imshow("Overlay Preview", overlay)

            # if s > 0.95 and s<1.05 and 
            # if ang >= -10.0 and ang <= 10.0:
            #     print (f"Angle: {ang},     Chamfer: {min_val    }")
            #     cv2.waitKey(0)  # pause for inspection
                
            k = cv2.waitKey(1) & 0xFF
            if k in {ord('q'), 27}:
                print("\nAbort requested – keeping current best.")
                scales = []      # break outer loop
                break
        else:
            continue
        break  # inner loop broke via 'q'
    
    fig, ax = plt.subplots()
    artist_scale_dict = {}
    for s_, chdict in chamfers.items():                    # draw every scale as points
        angs   = np.fromiter(chdict.keys(), dtype=float)
        costs  = np.fromiter((v[0] if isinstance(v, tuple) else v   # v = (cost, loc)
                            for v in chdict.values()), dtype=float)
        artist = ax.scatter(angs, costs, s=20, label=f"s={s_:0.2f}", picker=True)
        artist_scale_dict[artist] = s_
    ax.set_xlabel("Rotation angle (°)")
    ax.set_ylabel("Chamfer cost")
    ax.legend()
    fig.tight_layout()


    # -----  interactive callback ----------------------------------------------------
    def on_click(event):
        """Callback: user clicks anywhere on the plot → show overlay for nearest pt."""

        s = artist_scale_dict[event.artist]
        ang = list(chamfers[s].keys())[event.ind[0]]
        # single warp that includes *both* rotation and scale
        M_rs = cv2.getRotationMatrix2D((cx, cy), ang, s)
        rotated = cv2.warpAffine(moving_bw, M_rs, (w, h),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # translation search via Chamfer convolution
        recall_score_map = chamfer_score(recall_dist, rotated)
        accuracy_score_map = chamfer_score(accuracy_dist, rotated)
        # score_map = recall_score_map / (accuracy_score_map + 1e-9)
        score_map = recall_score_map / (accuracy_score_map + 1e-9) ** (1/3)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(score_map)

        chamfers[s][ang] = max_val
        # ─── live visualisation ───
        cost_vis = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX)
        cost_vis = cv2.applyColorMap(cost_vis.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.circle(cost_vis, max_loc, 4, (0, 0, 255), -1)
        cv2.putText(cost_vis, f"rot {ang:.1f}  s {s:.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow("Chamfer Cost", cost_vis)

        M_tmp = M_rs.copy()
        M_tmp[0, 2] += max_loc[0] - cx
        M_tmp[1, 2] += max_loc[1] - cy
        warped_edge = cv2.warpAffine(moving_bw * 255, M_tmp, (w, h),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        dist_uint = (recall_dist * 255 / recall_dist.max()).astype(np.uint8)
        overlay = cv2.applyColorMap(dist_uint, cv2.COLORMAP_TURBO)
        overlay[fixed_bw > 0] = (0, 0, 0)
        # cnt, _ = cv2.findContours(warped_edge, cv2.RETR_EXTERNAL,
        #                           cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(overlay, cnt, -1, (0, 255, 0), 1)
        overlay[warped_edge > 0] = (0, 0, 255)
        cv2.imshow("Overlay Preview", overlay)
        cv2.waitKey(1)
        print(f"[click angle={ang:.1f}  scale={s:.3f}  cost={max_val:.0f}")

    # connect callback ↔︎ figure
    cid = fig.canvas.mpl_connect('pick_event', on_click)     # :contentReference[oaicite:2]{index=2}
    plt.show()                                            # keep UI reactive
    # plt.pause(0.001) 
    # ------------------------------------------------------------------------------

    # while cv2.waitKey(1) != 27:      # 4. keep the program running
    #     pass


    print("\n>>> Best pose found")
    print(f"   scale        : {best['scale']:.3f}")
    print(f"   rotation     : {best['angle']:.2f}°")
    print(f"   translation  : dx={best['dx']:.1f}px  dy={best['dy']:.1f}px")
    print(f"   Chamfer cost : {best['score']:.0f}")

    # final warp
    fac = mpp_fixed / args_mpp
    M_final = cv2.getRotationMatrix2D((cx, cy), best['angle'], best['scale'])
    M_final[0, 2] += best['dx']
    M_final[1, 2] += best['dy']

    # M_final[0, 2] *= fac
    # M_final[1, 2] *= fac

    aligned = cv2.warpAffine(moving_bw * 255, M_final, (w, h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.destroyAllWindows()
    return M_final, aligned

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True, type=Path)
    ap.add_argument("--mpp", required=True, type=float)
    ap.add_argument("--cloud", required=True, type=Path)
    ap.add_argument("--rot-range", nargs=2, type=float, default=(-180, 180),
                    metavar=("MIN_DEG", "MAX_DEG"))
    ap.add_argument("--rot-step", type=float, default=1.0)
    ap.add_argument("--scale-range", nargs=2, type=float, default=(0.8, 1.4),
                    metavar=("SMIN", "SMAX"),
                    help="uniform scale factors to scan (default 0.95 1.05)")
    ap.add_argument("--scale-step", type=float, default=0.02)
    ap.add_argument("--out", default="aligned_cloud.png", type=Path)
    args = ap.parse_args()

    M_final, aligned = align_cloud(args.fixed, args.mpp, args.cloud, 
                        rot_range=args.rot_range, rot_step=args.rot_step,
                        scale_range=args.scale_range, scale_step=args.scale_step)
    cv2.imwrite(str(args.out), aligned)
    print(f"Aligned mask written → {args.out}")

