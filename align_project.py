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
    
    final_transform_3D, aligned_img = align_cloud(floorplan_path, align_info["mpp"], cloud_path)
    cv2.imwrite(str(aligned_floorplan_path), aligned_img)
    align_info["scan_to_floorplan"] = final_transform_3D.tolist()  # Convert to list for JSON serialization

    # Save the updated alignment info
    with open(align_json, "w") as f:
        json.dump(align_info, f, indent=4)

    # Save final_transform_3D as dense_floorplan_transform.npy
    np.save(proj_dir / "floorplan" / "dense_floorplan_transform.npy", final_transform_3D)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align point cloud with floor plan")
    parser.add_argument("project_output_folder", type=Path, help="Path to the pipeline output folder")
    args = parser.parse_args()
    align_project(args.project_output_folder)