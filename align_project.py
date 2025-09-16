from align_cloud import align_cloud
import argparse
from pathlib import Path
import json
import cv2
import numpy as np

def extract_scale_from_transform(T):
    """Extract uniform scale from a 4x4 transform matrix"""
    R = T[:3, :3]
    scale_x = np.linalg.norm(R[0])
    scale_y = np.linalg.norm(R[1])
    scale_z = np.linalg.norm(R[2])
    uniform_scale = (scale_x + scale_y + scale_z) / 3.0
    return uniform_scale

def align_project(proj_dir: Path, wall_mask_path: Path = None):
    #Check if the project directory exists
    if not proj_dir.exists():
        raise FileNotFoundError(f"Project directory {proj_dir} does not exist.")
    
    align_json = proj_dir / "floorplan" / "floorplan_align_info.json"
    if align_json.exists():
        with open(align_json, "r") as f:
            align_info = json.load(f)
    else:
        print("Use align_cloud.py instead...")
        raise FileNotFoundError(f"Alignment info file {align_json} does not exist.")
    
    floorplan_path = proj_dir / "floorplan" / align_info["name"]
    aligned_floorplan_path = proj_dir / "floorplan" / "auto_aligned_floorplan.png"
    cloud_path = proj_dir / "vtour" / "MVSFormer_model.ply"
    if not cloud_path.exists():
        print("Pipeline hasn't run yet, so no point cloud to align with floorplan.")
        raise FileNotFoundError(f"Point cloud file {cloud_path} does not exist.")
    
    # Use provided wall mask or default path
    if wall_mask_path is None:
        waffle_mask_path = proj_dir / "floorplan" / "wall_mask_waffle.png"
    else:
        waffle_mask_path = wall_mask_path
    
    # Check if wall mask exists
    if not waffle_mask_path.exists():
        raise FileNotFoundError(f"Wall mask file not found: {waffle_mask_path}. Please ensure wall mask is created before running alignment.")
    final_transform_3D, aligned_img = align_cloud(waffle_mask_path, align_info["mpp"], cloud_path, debug=False)
    cv2.imwrite(str(aligned_floorplan_path), aligned_img)
    align_info["scan_to_floorplan"] = final_transform_3D.tolist()  # Convert to list for JSON serialization
    align_info["scale"] = extract_scale_from_transform(final_transform_3D)
    # Save the updated alignment info
    with open(align_json, "w") as f:
        json.dump(align_info, f, indent=4)

    # Save final_transform_3D as dense_floorplan_transform.npy
    np.save(proj_dir / "floorplan" / "dense_floorplan_transform.npy", final_transform_3D)
    np.savetxt(proj_dir / "floorplan" / "dense_floorplan_scale.txt", [align_info["scale"]], fmt='%.6f')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align point cloud with floor plan")
    parser.add_argument("project_output_folder", type=Path, help="Path to the pipeline output folder")
    parser.add_argument("--wall_mask", type=Path, help="Path to wall mask file (if not provided, will use wall_mask_waffle.png from floorplan directory)")
    args = parser.parse_args()
    align_project(args.project_output_folder, args.wall_mask)