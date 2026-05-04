# Pipeline Design

This document describes the full pipeline from raw scanner output to IFC model, and serves as the design reference for the IFC builder module.

---

## End-to-end pipeline

```
┌─────────────────────────────────────────────┐
│  ACQUISITION                                │
│  Slamtec Aurora S  →  scan.ply              │
│  (or RGBD / LiDAR / monocular video)        │
└────────────────────┬────────────────────────┘
                     │ .ply (axis-aligned, Z up)
                     ▼
┌─────────────────────────────────────────────┐
│  PREPROCESSING  (preprocess_for_spatiallm.py│
│  • Statistical outlier removal (3 modes)    │
│  • DBSCAN largest-cluster filtering         │
│  • Z-up alignment via PCA                   │
│  • Manhattan alignment (walls → X/Y axes)   │
│  • Metric scaling (target height = 2.5 m)   │
│  • Optional voxel downsampling              │
│  • Optional color normalization             │
└────────────────────┬────────────────────────┘
                     │ clean, axis-aligned .ply
                     ▼
┌─────────────────────────────────────────────┐
│  SPATIALLM INTERNAL PREP  (inference.py)    │
│  • Voxel grid sampling (GridSample)         │
│  • Color normalization                      │
│  • Coord discretization to integer bins     │
└────────────────────┬────────────────────────┘
                     │ point cloud tensor
                     ▼
┌─────────────────────────────────────────────┐
│  SPATIALLLM INFERENCE  (inference.py)       │
│  • Sonata encoder → point tokens            │
│  • LLM (Qwen-0.5B or Llama-1B) decoding    │
│  • Structured layout text generation        │
└────────────────────┬────────────────────────┘
                     │ layout .txt
                     ▼
┌─────────────────────────────────────────────┐
│  IFC BUILDER  (ifc_builder.py — WIP)        │
│  • Parse layout entities                    │
│  • Map to IFC schema                        │
│  • Write .ifc file                          │
└────────────────────┬────────────────────────┘
                     │ .ifc
                     ▼
           BIM tools (Revit, ArchiCAD,
           BlenderBIM, FreeCAD…)
```

---

## Preprocessing script

`preprocess_for_spatiallm.py` prepares a raw PLY point cloud for SpatialLM inference. It runs a fixed sequence of steps and writes a clean PLY file that can be fed directly to `inference.py`.

### Processing steps (in order)

| Step | Function | What it does |
|---|---|---|
| 1. Denoise | `denoise()` | Removes statistical outliers; optionally adds a radius pass |
| 2. Cluster filter | `keep_largest_cluster()` | Discards fragments by keeping only the DBSCAN largest cluster |
| 3. Z-up alignment | `align_z_up_pca()` | Rotates the cloud so the gravity axis is Z; tries RANSAC floor detection on 4 overlapping bottom slices (0–15 %, 0–25 %, 0–10 %, 5–20 %) and falls back to PCA if the best horizontal score is below 0.85 |
| 4. Manhattan alignment | `align_manhattan()` | Rotates around Z so the dominant wall direction is parallel to X/Y axes — required by SpatialLM |
| 5. Metric scaling | `scale_to_metric()` | Rescales the cloud so the floor-to-ceiling height equals `--target_height` (default 2.5 m) |
| 6. Voxel downsample | built-in | Optional uniform downsampling via `--voxel_size` |
| 7. Color normalization | `normalize_colors()` | Shifts color distribution to match SpatialLM training statistics |

### Denoising modes

| Mode | Passes | Radius pass | Use when |
|---|---|---|---|
| `conservative` | 1 × (20 neighbors, σ=2.5) | No | Dense, high-quality scans; preserves fine details |
| `moderate` (default) | 1 × (20 neighbors, σ=2.0) | No | General use |
| `aggressive` | 2 × (20→10 neighbors, σ=2.0→1.5) + radius | Yes | Noisy / sparse scans |

### Usage

```bash
# Single file, default mode
python preprocess_for_spatiallm.py -i noisy.ply -o clean.ply

# Conservative denoising
python preprocess_for_spatiallm.py -i noisy.ply -o clean.ply --mode conservative

# Batch: all .ply files in a folder
python preprocess_for_spatiallm.py -i scans/ -o clean_scans/ --mode aggressive

# Aggressive + keep largest cluster + voxel downsample + color normalization
python preprocess_for_spatiallm.py -i noisy.ply -o clean.ply \
    --mode aggressive \
    --keep_largest_cluster \
    --voxel_size 0.02 \
    --normalize_colors
```

**All flags**

| Flag | Default | Description |
|---|---|---|
| `-i / --input` | — | Input `.ply` file or folder of `.ply` files |
| `-o / --output` | — | Output `.ply` file or folder |
| `--mode` | `moderate` | Denoising preset: `conservative` / `moderate` / `aggressive` |
| `--nb_neighbors` | — | Custom outlier filter neighbor count (pair with `--std_ratio`) |
| `--std_ratio` | — | Custom outlier filter std multiplier |
| `--use_radius` | off | Add a radius-based outlier pass on top of the statistical one |
| `--radius` | `0.05` | Radius for the radius pass (metres) |
| `--radius_min_points` | `8` | Minimum neighbors within radius |
| `--keep_largest_cluster` | off | Run DBSCAN and drop all but the largest cluster |
| `--dbscan_eps` | `0.05` | DBSCAN neighborhood radius |
| `--dbscan_min_points` | `20` | DBSCAN minimum cluster size |
| `--no_align` | off | Skip Z-up + Manhattan alignment |
| `--no_scale` | off | Skip metric scaling |
| `--target_height` | `2.5` | Floor-to-ceiling target height (metres) |
| `--voxel_size` | `0.0` | Voxel size for downsampling; `0` = disabled |
| `--normalize_colors` | off | Shift color statistics to match SpatialLM training distribution |

---

## Coordinate system

SpatialLM works in a right-handed coordinate system where **Z is up**. Coordinates in the layout text are discretized integers; `Layout.undiscretize_and_unnormalize()` converts them back to metric values (meters).

The Slamtec Aurora S exports point clouds in its own sensor frame. A pre-processing step must:
1. Rotate the cloud so Z is the gravity-up axis.
2. Optionally translate the origin to a known reference point (e.g. room corner or floor level).

---

## Layout entity → IFC mapping

### Wall → `IfcWall`

```
Wall(ax, ay, az, bx, by, bz, height, thickness)
```

| Layout field | IFC / geometry meaning |
|---|---|
| `(ax, ay, az)` | Start point of wall centre-line (metres) |
| `(bx, by, bz)` | End point of wall centre-line (metres) |
| `height` | Wall height (metres) |
| `thickness` | Wall thickness (metres) |

IFC mapping:
- `IfcWall` with `IfcShapeRepresentation` → `IfcExtrudedAreaSolid`
- The wall extrusion profile is a rectangle of `thickness × height`
- Placed at start point, oriented along `(bx-ax, by-ay, bz-az)`

### Door → `IfcDoor`

```
Door(wall_id, position_x, position_y, position_z, width, height)
```

- `wall_id` links the door to its host wall
- `position_{x,y,z}` is the door sill centre in world coordinates
- IFC: `IfcDoor` + `IfcRelVoidsElement` on the host `IfcWall`

### Window → `IfcWindow`

```
Window(wall_id, position_x, position_y, position_z, width, height)
```

Same structure as Door.
- IFC: `IfcWindow` + `IfcRelVoidsElement` on the host `IfcWall`

### Furniture bbox → `IfcFurnishingElement`

```
Bbox(class, position_x, position_y, position_z, angle_z, scale_x, scale_y, scale_z)
```

| Layout field | IFC / geometry meaning |
|---|---|
| `class` | Object category string (e.g. `"sofa"`, `"bed"`) |
| `position_{x,y,z}` | Bounding box centre in world coordinates |
| `angle_z` | Rotation around Z-axis (radians) |
| `scale_{x,y,z}` | Full extents of the bounding box (metres) |

IFC mapping:
- `IfcFurnishingElement` with a box-shaped `IfcShapeRepresentation`
- `IfcLocalPlacement` encodes position and Z-rotation
- `Pset_FurnishingElementCommon.Reference` stores the category label

---

## IFC file structure (planned)

```
IfcProject
  └─ IfcSite
       └─ IfcBuilding
            └─ IfcBuildingStorey   (one per detected floor level)
                 ├─ IfcSpace       (detected room, optional)
                 ├─ IfcWall × N
                 │    ├─ IfcDoor (via IfcRelVoidsElement)
                 │    └─ IfcWindow (via IfcRelVoidsElement)
                 └─ IfcFurnishingElement × M
```

---

## IFC library

[IfcOpenShell](https://ifcopenshell.org/) (`ifcopenshell` Python package) will be used to write IFC files.

```python
import ifcopenshell
import ifcopenshell.api

model = ifcopenshell.file()
# ... build hierarchy and geometry
model.write("output.ifc")
```

Install:
```bash
pip install ifcopenshell
```

---

## Multi-room / multi-floor (future)

When scanning more than one room or floor:
1. Each room is scanned separately and produces one PLY file.
2. Each PLY is processed independently by SpatialLM.
3. A merge step aligns room coordinate frames (using shared reference markers or ICP) and combines them into a single `IfcBuilding` with multiple `IfcBuildingStorey` elements.

---

## References

- [IFC 4.3 specification](https://ifc43-docs.standards.buildingsmart.org/)
- [IfcOpenShell documentation](https://docs.ifcopenshell.org/)
- [SpatialLM paper](https://arxiv.org/abs/2506.07491)
- [Slamtec Aurora S product page](https://www.slamtec.com/en/Lidar/AuroraS)
