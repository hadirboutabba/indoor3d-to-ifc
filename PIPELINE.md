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
│  PREPROCESSING  (inference.py)              │
│  • Outlier removal (cleanup_pcd)            │
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
