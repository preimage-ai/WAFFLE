from align_cloud import align_cloud
import argparse
from pathlib import Path
import json
import cv2
import numpy as np

def align_project(proj_dir: Path):
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
    
    affine2D, aligned_img = align_cloud(floorplan_path, align_info["mpp"], cloud_path,
                        rot_range=args.rot_range, rot_step=args.rot_step,
                        scale_range=args.scale_range, scale_step=args.scale_step)
    cv2.imwrite(str(aligned_floorplan_path), aligned_img)
    scale_x = np.linalg.norm(affine2D[0, :2])
    scale_y = np.linalg.norm(affine2D[1, :2])
    scale = (scale_x + scale_y) / 2.0  # Average scale to scale z by the same factor
    translation_2d = affine2D[:, 2]

    # Convert 2D rotation to 3D (assuming rotation around the z-axis)
    rotation_3d = np.eye(3)
    rotation_3d[0, 0] = affine2D[0, 0]
    rotation_3d[0, 1] = affine2D[0, 1]
    rotation_3d[1, 0] = affine2D[1, 0]
    rotation_3d[1, 1] = affine2D[1, 1]
    rotation_3d[2, 2] = scale # z-axis scaled by the avg scale that x and y went through 

    # Build a full 4x4 transformation matrix (homogeneous)
    transform_3d = np.eye(4)
    transform_3d[:3, :3] = rotation_3d
    transform_3d[:3, 3] = [translation_2d[0], translation_2d[1], 0]  # z-axis translation set to zero if points are in xy
    align_info["scan_to_floorplan"] = transform_3d.tolist()  # Convert to list for JSON serialization

    # Save the updated alignment info
    with open(align_json, "w") as f:
        json.dump(align_info, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align point cloud with floor plan")
    parser.add_argument("project_output_folder", required=True, type=Path, help="Path to the pipeline output folder")
    args = parser.parse_args()
    align_project(args.project_output_folder)