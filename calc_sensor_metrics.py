import open3d as o3d
import numpy as np
import os

def calculate_sensor_metrics(pcd_path, label):
    print(f"\n--- Processing {label} ---")
    
    # SAFETY CHECK 1: Verify the file actually exists
    if not os.path.exists(pcd_path):
        print(f"❌ ERROR: File not found at '{pccd_path}'")
        print("   Please check the path and ensure the file exists.")
        return None, None, None

    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # SAFETY CHECK 2: Verify the file actually loaded and has points
    if len(pcd.points) == 0:
        print(f"❌ ERROR: Point cloud is empty. The file might be corrupted or in an unsupported format.")
        return None, None, None
        
    print(f"✅ Successfully loaded {len(pcd.points):,} points.")
    points = np.asarray(pcd.points)
    
    # 1. Point Density (pts/m²) on a standard vertical wall
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    wall_pcd = pcd.select_by_index(inliers)
    wall_points = np.asarray(wall_pcd.points)
    
    normal = np.array(plane_model[:3])
    drop_axis = np.argmax(np.abs(normal))
    keep_axes = [i for i in range(3) if i != drop_axis]
    wall_2d = wall_points[:, keep_axes]
    
    min_2d = np.min(wall_2d, axis=0)
    max_2d = np.max(wall_2d, axis=0)
    wall_area = (max_2d[0] - min_2d[0]) * (max_2d[1] - min_2d[1])
    
    point_density = len(wall_points) / wall_area if wall_area > 0 else 0
    print(f"Point Density: {point_density:.1f} pts/m²")

    # 2. Surface Roughness (σ in mm)
    a, b, c, d = plane_model
    distances = np.abs(a * wall_points[:, 0] + b * wall_points[:, 1] + c * wall_points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
    roughness_mm = np.std(distances) * 1000  # Convert meters to mm
    print(f"Surface Roughness: {roughness_mm:.2f} mm")

    # 3. Completeness (%)
    voxel_size = 0.05
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    occupied_voxels = len(downsampled_pcd.points)
    
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    total_voxels_in_bbox = np.prod(np.ceil(bbox_extent / voxel_size))
    
    completeness = (occupied_voxels / total_voxels_in_bbox) * 100
    print(f"Completeness: {completeness:.1f}%")

    return point_density, roughness_mm, completeness

# --- MAIN EXECUTION ---
# ⚠️ ⚠️ ⚠️ UPDATE THESE PATHS ⚠️ ⚠️ ⚠️
# Use the FULL absolute path in WSL format to avoid any issues.
# Example: "/mnt/c/Users/hboutabb/SpatialLM/data/aurora_k001.ply"

aurora_path = "/mnt/c/Users/hboutabb/SpatialLM/cuisine_aurora.ply"   # <-- CHANGE THIS
trimble_path = "/mnt/c/Users/hboutabb/SpatialLM/cuisine_trimble.ply" # <-- CHANGE THIS

print("Calculating Sensor Metrics for Table 8...")
den_a, rough_a, comp_a = calculate_sensor_metrics(aurora_path, "Aurora S + RPLIDAR")
den_t, rough_t, comp_t = calculate_sensor_metrics(trimble_path, "Trimble X7 (TLS)")

if den_a is not None and den_t is not None:
    print("\n=== COPY THESE TO TABLE 8 ===")
    print(f"Aurora S   | Density: {den_a:.0f} | Roughness: {rough_a:.1f} | Completeness: {comp_a:.1f}")
    print(f"Trimble X7 | Density: {den_t:.0f} | Roughness: {rough_t:.1f} | Completeness: {comp_t:.1f}")
else:
    print("\n⚠️ Script aborted due to file loading errors. Please fix the paths above and try again.")