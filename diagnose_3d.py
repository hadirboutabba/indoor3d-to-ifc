import open3d as o3d
import numpy as np

# --- UPDATE THESE PATHS TO YOUR ACTUAL FILES ---
trimble_path = "cuisine_trimble.ply"      # Your Trimble X7 point cloud
ifc_obj_path = "obj.obj"    # Your generated IFC exported as .obj

print("--- DIAGNOSTIC: Checking Scales and Origins ---\n")

# 1. Load Trimble X7
trimble_pcd = o3d.io.read_point_cloud(trimble_path)
t_bbox = trimble_pcd.get_axis_aligned_bounding_box()
t_min = np.asarray(t_bbox.get_min_bound())
t_max = np.asarray(t_bbox.get_max_bound())
print(f"Trimble X7 Points: {len(trimble_pcd.points):,}")
print(f"Trimble X7 Min Bound (X,Y,Z): {t_min}")
print(f"Trimble X7 Max Bound (X,Y,Z): {t_max}")
print(f"Trimble X7 Size (Extent): {t_max - t_min}\n")

# 2. Load Generated IFC
gen_mesh = o3d.io.read_triangle_mesh(ifc_obj_path)
g_bbox = gen_mesh.get_axis_aligned_bounding_box()
g_min = np.asarray(g_bbox.get_min_bound())
g_max = np.asarray(g_bbox.get_max_bound())
print(f"Generated IFC Vertices: {len(gen_mesh.vertices):,}")
print(f"Generated IFC Min Bound (X,Y,Z): {g_min}")
print(f"Generated IFC Max Bound (X,Y,Z): {g_max}")
print(f"Generated IFC Size (Extent): {g_max - g_min}\n")

# 3. Compare
size_ratio = (t_max - t_min) / (g_max - g_min + 1e-6)
print(f"Size Ratio (Trimble / Generated): {size_ratio}")