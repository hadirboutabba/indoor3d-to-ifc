import open3d as o3d
import numpy as np

# 1. Load the Trimble X7 Point Cloud (The Ground Truth Reality)
trimble_pcd = o3d.io.read_point_cloud("cuisine_trimble.ply")

# 2. Load your Generated IFC (Exported as .obj or .ply from BlenderBIM/FreeCAD)
generated_mesh = o3d.io.read_triangle_mesh("obj.obj")
generated_pcd = generated_mesh.sample_points_uniformly(number_of_points=100000)

# 3. Calculate Chamfer Distance (How far are your generated walls from the real laser scan?)
distances = np.asarray(generated_pcd.compute_point_cloud_distance(trimble_pcd))
chamfer_distance_mm = np.mean(distances) * 1000

print(f"Geometric Deviation (Chamfer Distance): {chamfer_distance_mm:.2f} mm")