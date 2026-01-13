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
    return cv2.filter2D(dist_map, cv2.CV_32F, tmpl, borderType=cv2.BORDER_REPLICATE)


def render_pointcloud(cloud_path: str,
                      mpp: float,
                      background=(0, 0, 0, 0),
                      color_by="height",
                      out_path: str | None = None):
    """
    Orthographic bird's-eye render of a point cloud using Open3D.

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

    DOWNSAMPLE_VOXEL_SIZE = 0.02

    # ------------------------------------------------------------
    # 1. Load point cloud and basic stats
    # ------------------------------------------------------------
    pcd = o3d.t.io.read_point_cloud(cloud_path)
    
    seg = pcd.point.seg.numpy().reshape(-1)
    normals = pcd.point.normals.numpy()
    
    mask = np.ones_like(seg).astype(np.bool_)
    mask = np.logical_and(mask, np.logical_or(seg == 0, seg == 1, seg == 32))  # wall, building, fence
    mask = np.logical_and(mask, np.abs(normals[:, 2]) < 0.3)
    
    pcd = o3d.io.read_point_cloud(cloud_path)
    pts = np.asarray(pcd.points)[mask]
    pcd.points = o3d.utility.Vector3dVector(pts)
    if pts.size == 0:
        raise ValueError("Point cloud is empty!")
    
    # Downsample to reduce GPU load and avoid segfault
    pcd = pcd.voxel_down_sample(voxel_size=DOWNSAMPLE_VOXEL_SIZE)
    pts = np.asarray(pcd.points)

    # XY extent → pixel resolution
    min_xy = pts[:, :2].min(0)
    max_xy = pts[:, :2].max(0)
    extent_xy = max_xy - min_xy
    width_px = int(np.ceil(extent_xy[0] / mpp))
    height_px = int(np.ceil(extent_xy[1] / mpp))

    # ------------------------------------------------------------
    # 2. Prepare colours (height, intensity or original RGB)
    # ------------------------------------------------------------
    colours = np.ones_like(pts)    # light grey fallback
    pcd.colors = o3d.utility.Vector3dVector(colours)

    # ------------------------------------------------------------
    # 3. Build an off-screen scene and orthographic camera
    # ------------------------------------------------------------
    renderer = o3d.visualization.rendering.OffscreenRenderer(width_px, height_px)
    scene = renderer.scene
    scene.set_background(background)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 1.0
    scene.add_geometry("cloud", pcd, mat)

    # 3.2 Camera: look straight down –Z  (world Z=up convention)
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent_z = bbox.get_extent()[2]
    eye = center + np.array([0, 0, extent_z * 2])
    up = np.array([0, 1, 0])

    cam = scene.camera
    cam.set_projection(o3d.visualization.rendering.Camera.Projection.Ortho,
                       -extent_xy[0]/2, extent_xy[0]/2,
                       -extent_xy[1]/2, extent_xy[1]/2,
                       0.1, extent_z*4)
    cam.look_at(center, eye, up)

    # ------------------------------------------------------------
    # 4. Render
    # ------------------------------------------------------------
    img_o3d = renderer.render_to_image()
    img_np = np.asarray(img_o3d)

    # optional disk write
    if out_path:
        o3d.io.write_image(out_path, img_o3d)

    cam_dict = dict(width=width_px, height=height_px,
                    left=-extent_xy[0]/2, right=extent_xy[0]/2,
                    bottom=-extent_xy[1]/2, top=extent_xy[1]/2,
                    near=0.1, far=extent_z*4,
                    eye=eye.tolist(), center=center.tolist(), 
                    extent=extent_xy.tolist(),
                    up=up.tolist())
    
    _, bw = cv2.threshold(img_np.mean(axis=-1).astype(np.uint8), 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # import matplotlib.pyplot as plt
    # plt.imshow(bw)
    # plt.show()

    transform_mat = np.eye(4, dtype=np.float32)
    shift_x = center[0] - extent_xy[0]/2
    shift_y = center[1] + extent_xy[1]/2
    transform_mat[0, 3] = -shift_x
    transform_mat[1, 3] = -shift_y

    y_inv = np.eye(4, dtype=np.float32)
    y_inv[1, 1] = -1
    y_inv[:3, :3] = y_inv[:3, :3] / mpp   # Invert y axis when converting from cloud to image
    transform_mat = y_inv @ transform_mat

    return bw, cam_dict, transform_mat


def downscale_img(img, mpp, side=512):
    tgtsize = side * side
    imgsize = img.shape[0] * img.shape[1]
    fac = np.sqrt(tgtsize / imgsize)
    img_resized = cv2.resize(img, (int(img.shape[1] * fac), int(img.shape[0] * fac)), interpolation=cv2.INTER_LINEAR)
    mpp_new = mpp / fac
    return img_resized, mpp_new

def pad_for_rotation(img: np.ndarray) -> np.ndarray:
    """Pad image to be square so that the rotation of the image does not crop it."""
    h, w = img.shape
    diag = int(np.ceil(np.sqrt(h**2 + w**2)))
    pad_h = (diag - h) // 2
    pad_w = (diag - w) // 2
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                cv2.BORDER_CONSTANT, value=0)
    pad_transform = np.eye(4, dtype=np.float32)
    pad_transform[0, 3] = pad_w
    pad_transform[1, 3] = pad_h
    return padded, pad_transform

# ──────────────── main ────────────────
def align_cloud(fixed: Path, args_mpp: float, cloud: Path, rot_range = (-180, 180), rot_step = 1.0, scale_range = (0.8, 1.4), scale_step = 0.02, debug=False):
    mpp = 0.05
    moving_bw, cam_info, cloud_shift = render_pointcloud(str(cloud), mpp, out_path="test.png")

    fixed_bw = load_binary(fixed)
    floorplan_height = fixed_bw.shape[0] * args_mpp
    fixed_bw, mpp_fixed = downscale_img(fixed_bw, mpp=args_mpp)

    fac = mpp / mpp_fixed
    moving_bw = cv2.resize(moving_bw, (int(moving_bw.shape[1] * fac), int(moving_bw.shape[0] * fac)), interpolation=cv2.INTER_LINEAR)
    scaling_mat = np.eye(4, dtype=np.float32)
    scaling_mat[:3, :3] = scaling_mat[:3, :3] * fac
    cloud_shift = scaling_mat @ cloud_shift  # scale cloud-to-img transform matrix

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

    moving_bw, pad_trans = pad_for_rotation(moving_bw)
    mh, mw = moving_bw.shape
    cx, cy = mw / 2.0, mh / 2.0
    cloud_shift = pad_trans @ cloud_shift  # add padding shift in cloud-to-img transform matrix
    
    if debug:
        cv2.namedWindow("tst", cv2.WINDOW_NORMAL)
        cv2.imshow("tst", (moving_bw * 255).astype(np.uint8))
        cv2.waitKey(0)

    if debug:
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
            c_shift_x = mw * (s - 1) / 2    # shift in x to center the kernel img
            c_shift_y = mh * (s - 1) / 2    # shift in y to center the kernel img
            M_rs[0, 2] += c_shift_x
            M_rs[1, 2] += c_shift_y
            rotated = cv2.warpAffine(moving_bw, M_rs, (int(mw * s), int(mh * s)),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if debug:
                cv2.imshow("tst", (rotated * 255).astype(np.uint8))

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
            if debug:
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
                M_tmp[0, 2] += max_loc[0] - cx - c_shift_x
                M_tmp[1, 2] += max_loc[1] - cy - c_shift_y
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
    
    if debug:
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
        c_shift_x = mw * (s - 1) / 2    # shift in x to center the kernel img
        c_shift_y = mh * (s - 1) / 2    # shift in y to center the kernel img
        M_rs[0, 2] += c_shift_x
        M_rs[1, 2] += c_shift_y
        rotated = cv2.warpAffine(moving_bw, M_rs, (int(mw * s), int(mh * s)),
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
        M_tmp[0, 2] += max_loc[0] - cx - c_shift_x
        M_tmp[1, 2] += max_loc[1] - cy - c_shift_y
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

        def on_cv2_click(event, x, y, flags, params):
            """Callback: user clicks on the OpenCV window → print current pose."""
            if event == cv2.EVENT_LBUTTONDOWN:
                M_tmp = M_rs.copy()
                M_tmp[0, 2] += x - cx
                M_tmp[1, 2] += y - cy
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
        
        cv2.setMouseCallback("Chamfer Cost", on_cv2_click)
        while cv2.waitKey(1) & 0xFF != ord('q'):
            pass

        print(f"[click angle={ang:.1f}  scale={s:.3f}  cost={max_val:.0f}")

    # connect callback ↔︎ figure
    if debug:
        cid = fig.canvas.mpl_connect('pick_event', on_click)     # :contentReference[oaicite:2]{index=2}
        plt.show()                                            # keep UI reactive


    print("\n>>> Best pose found")
    print(f"   scale        : {best['scale']:.3f}")
    print(f"   rotation     : {best['angle']:.2f}°")
    print(f"   translation  : dx={best['dx']:.1f}px  dy={best['dy']:.1f}px")
    print(f"   Chamfer cost : {best['score']:.0f}")

    # final warp
    fac = mpp_fixed / args_mpp
    M_img = cv2.getRotationMatrix2D((cx, cy), best['angle'], best['scale'])
    M_img[0, 2] += best['dx']
    M_img[1, 2] += best['dy']

    aligned = cv2.warpAffine(moving_bw * 255, M_img, (w, h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    affinemat = np.eye(4, dtype=np.float32)
    affinemat[:2, :2] = M_img[:2, :2]
    affinemat[:2, 3] = M_img[:2, 2]
    affinescale = np.linalg.det(M_img[:2, :2]) ** (1/2)  # scale factor
    affinemat[2, 2] = affinescale
    M_cloud = affinemat @ cloud_shift
    y_inv = np.eye(4, dtype=np.float32) # Invert y axis when converting from image to cloud
    y_inv[1, 1] = -1
    y_inv[:3, :3] = y_inv[:3, :3] * mpp_fixed
    M_cloud = y_inv @ M_cloud
    # move cloud in y by floorplan height, due to y flip and cloud origin shift final cloud seems to w origin at top left corner
    # so we need to move it down by the floorplan height
    T_y = np.eye(4, dtype=np.float32)
    T_y[1, 3] = floorplan_height
    M_cloud = T_y @ M_cloud

    if debug:
        cv2.destroyAllWindows()
    
    return M_cloud, aligned

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
                        scale_range=args.scale_range, scale_step=args.scale_step, debug=True)
    cv2.imwrite(str(args.out), aligned)
    print(f"Aligned mask written → {args.out}")

    print ("Final transformation matrix:")
    print (M_final)