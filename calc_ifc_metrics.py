import open3d as o3d
import numpy as np
import ifcopenshell
import ifcopenshell.geom

def ifc_to_mesh(ifc_path):
    """Converts an IFC file to an Open3D TriangleMesh"""
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    iterator = ifcopenshell.geom.iterator(settings, ifcopenshell.open(ifc_path))
    meshes = []
    
    if iterator.initialize():
        while True:
            shape = iterator.get()
            geometry = shape.geometry
            verts = np.array(geometry.verts).reshape(-1, 3)
            faces = np.array(geometry.faces).reshape(-1, 3)
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            meshes.append(mesh)
            
            if not iterator.next():
                break
                
    # Combine all meshes into one
    if meshes:
        return sum(meshes[1:], meshes[0])
    return None

def calculate_chamfer_distance(mesh1, mesh2, num_samples=100000):
    """Calculates Chamfer Distance in mm"""
    pcd1 = mesh1.sample_points_uniformly(number_of_points=num_samples)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=num_samples)
    
    dist1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    dist2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    
    cd_meters = (np.mean(dist1) + np.mean(dist2)) / 2
    return cd_meters * 1000 # Return in mm

def calculate_voxel_iou(mesh1, mesh2, voxel_size=0.05):
    """Calculates Volumetric IoU using voxelization"""
    # Voxelize meshes by sampling their surfaces densely
    pcd1 = mesh1.sample_points_uniformly(number_of_points=500000)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=500000)
    
    vg1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1, voxel_size=voxel_size)
    vg2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2, voxel_size=voxel_size)
    
    voxels1 = set(tuple(v.grid_index) for v in vg1.voxels)
    voxels2 = set(tuple(v.grid_index) for v in vg2.voxels)
    
    intersection = len(voxels1.intersection(voxels2))
    union = len(voxels1.union(voxels2))
    
    return intersection / union if union > 0 else 0

def get_ifc_bboxes(ifc_path, element_type):
    """Extracts bounding boxes for specific IFC elements (Doors/Windows)"""
    f = ifcopenshell.open(ifc_path)
    elements = f.by_type(element_type)
    bboxes = []
    
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    
    for elem in elements:
        try:
            shape = ifcopenshell.geom.create_shape(settings, elem)
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            if len(verts) > 0:
                min_b = np.min(verts, axis=0)
                max_b = np.max(verts, axis=0)
                bboxes.append((min_b, max_b))
        except:
            continue
    return bboxes

def calculate_f1(pred_bboxes, gt_bboxes, iou_threshold=0.25):
    """Calculates F1 score based on 3D Bounding Box IoU"""
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    
    for pb in pred_bboxes:
        min_p, max_p = pb
        vol_p = np.prod(max_p - min_p)
        best_iou = 0
        
        for i, gb in enumerate(gt_bboxes):
            min_g, max_g = gb
            vol_g = np.prod(max_g - min_g)
            
            # Calculate 3D Intersection
            min_i = np.maximum(min_p, min_g)
            max_i = np.minimum(max_p, max_g)
            inter = np.prod(np.maximum(0, max_i - min_i))
            
            union = vol_p + vol_g - inter
            iou = inter / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
            
    fn = len(gt_bboxes) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# --- MAIN EXECUTION ---
# UPDATE THESE PATHS to your generated IFC and Ground Truth IFC for Room K-001
pred_ifc_path = "pred_k001.ifc"  # Your pipeline's output
gt_ifc_path = "gt_k001.ifc"      # Manually traced Trimble X7 IFC

print("Loading IFC files and generating meshes (this may take a minute)...")
pred_mesh = ifc_to_mesh(pred_ifc_path)
gt_mesh = ifc_to_mesh(gt_ifc_path)

print("\n--- Calculating Geometric Metrics ---")
# 1. Wall IoU (Approximated by full mesh voxel IoU for simplicity)
iou = calculate_voxel_iou(pred_mesh, gt_mesh)
print(f"Wall IoU: {iou:.2f}")

# 2. Chamfer Distance
cd = calculate_chamfer_distance(pred_mesh, gt_mesh)
print(f"Chamfer Distance: {cd:.1f} mm")

print("\n--- Calculating Semantic F1 Scores ---")
# 3. Door F1
pred_doors = get_ifc_bboxes(pred_ifc_path, "IfcDoor")
gt_doors = get_ifc_bboxes(gt_ifc_path, "IfcDoor")
door_f1 = calculate_f1(pred_doors, gt_doors)
print(f"Door F1: {door_f1:.2f}")

# 4. Window F1
pred_windows = get_ifc_bboxes(pred_ifc_path, "IfcWindow")
gt_windows = get_ifc_bboxes(gt_ifc_path, "IfcWindow")
window_f1 = calculate_f1(pred_windows, gt_windows)
print(f"Window F1: {window_f1:.2f}")

print("\n=== COPY THESE TO TABLE 9 ===")
print(f"IoU: {iou:.2f} | CD: {cd:.1f} mm | Door F1: {door_f1:.2f} | Window F1: {window_f1:.2f}")