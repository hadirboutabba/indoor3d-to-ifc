# Technical Documentation — Indoor3D-to-IFC

## Table of contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Layout TXT format](#2-layout-txt-format)
3. [preprocess_for_spatiallm.py](#3-preprocess_for_spatiallmpy)
4. [preprocess_large_building.py](#4-preprocess_large_buildingpy)
5. [inference.py](#5-inferencepy)
6. [postprocess.py](#6-postprocesspy)
7. [spatiallm_to_ifc.py](#7-spatiallm_to_ifcpy)
8. [run_pipeline.py](#8-run_pipelinepy)
9. [pipeline_gui.py](#9-pipeline_guipy)
10. [Docker](#10-docker)
11. [IFC output structure](#11-ifc-output-structure)
12. [Known limitations](#12-known-limitations)

---

## 1. Pipeline overview

```
raw scan (.ply)
      │
      ▼  preprocess_for_spatiallm.py  (or preprocess_large_building.py)
clean, axis-aligned cloud  (*_clean.ply)
      │
      ▼  inference.py
raw layout  (*_raw_layout.txt)
      │
      ▼  postprocess.py
corrected layout  (*_refined_layout.txt)
      │
      ▼  spatiallm_to_ifc.py
BIM model  (*.ifc)
```

Each step can be run independently. `run_pipeline.py` chains all four in a single command.
`pipeline_gui.py` exposes the same pipeline through a browser-based Streamlit interface.

---

## 2. Layout TXT format

All scripts read and write the same text format produced by SpatialLM inference.

**Schema**

```
wall_0=Wall(x1, y1, z1, x2, y2, z2, height, thickness)
door_0=Door(parent_wall, cx, cy, cz, width, height)
window_0=Window(parent_wall, cx, cy, cz, width, height)
bbox_0=Bbox(label, cx, cy, cz, rotation_rad, length, width, height)
```

**Parameter reference**

| Type | Parameter | Unit | Description |
|---|---|---|---|
| Wall | `x1,y1,z1` | m | Wall start point in world space |
| Wall | `x2,y2,z2` | m | Wall end point in world space |
| Wall | `height` | m | Wall height |
| Wall | `thickness` | m | Wall thickness (may be 0.0 — not always predicted by SpatialLM) |
| Door / Window | `parent_wall` | — | Name of the host wall (e.g. `wall_1`) |
| Door / Window | `cx,cy,cz` | m | Centre of the opening in world space |
| Door / Window | `width,height` | m | Opening dimensions |
| Bbox | `label` | — | Furniture category (e.g. `sofa`, `bed`, `desk`) |
| Bbox | `cx,cy,cz` | m | Object centre in world space |
| Bbox | `rotation_rad` | rad | Rotation around Z axis (counter-clockwise from X) |
| Bbox | `length,width,height` | m | Dimensions — length is the longest horizontal axis |

**Example**
```
wall_1=Wall(-0.436,-1.986,0.25, 1.789,-0.136,0.25, 2.14, 0.0)
window_0=Window(wall_1, 0.914,-0.786,1.55, 1.0, 1.02)
bbox_0=Bbox(chair,-0.386,-1.136,0.7, -0.785, 0.5625,0.515625,0.875)
```

---

## 3. preprocess_for_spatiallm.py

Prepares a raw `.ply` point cloud for SpatialLM inference: denoising, Z-up alignment,
Manhattan frame alignment, and metric scaling.

**Usage**
```bash
python preprocess_for_spatiallm.py \
  -i raw_scan.ply \
  -o clean_scan.ply \
  --mode moderate \
  --target_height 2.5 \
  --keep_largest_cluster \
  --normalize_colors
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `-i` / `--input` | required | Input `.ply` file or folder of `.ply` files |
| `-o` / `--output` | required | Output cleaned `.ply` file or folder |
| `--mode` | `moderate` | Denoising intensity: `conservative`, `moderate`, `aggressive` |
| `--target_height` | `2.5` | Target ceiling height in metres — used for metric scaling |
| `--no_denoise` | off | Skip statistical outlier removal |
| `--no_align` | off | Skip automatic axis alignment |
| `--keep_largest_cluster` | off | Discard everything except the largest DBSCAN cluster |
| `--normalize_colors` | off | Normalise the RGB distribution |
| `--voxel_size` | None | Voxel downsampling size in metres (e.g. `0.02`) |

**Denoising modes**

| Mode | Behaviour |
|---|---|
| `conservative` | One gentle statistical pass — preserves fine details |
| `moderate` | One standard statistical pass — good default |
| `aggressive` | Two passes + optional radius filter — best for noisy/sparse scans |

**Processing pipeline (per file)**

1. Load `.ply` with Open3D
2. Statistical outlier removal (one or two passes depending on mode)
3. Detect the floor plane with RANSAC (cascading Z-slices + PCA fallback) → rotate to Z-up
4. PCA on horizontal plane → rotate to align walls with X/Y axes (Manhattan frame)
5. Scale so that the detected floor-to-ceiling height matches `--target_height`
6. Optional: keep only the largest cluster, normalise colours, voxel downsample
7. Save cleaned `.ply`

---

## 4. preprocess_large_building.py

Variant of the preprocessor designed for large or multi-room buildings where the standard
script may struggle with memory or produce poor alignment.

**Usage**
```bash
python preprocess_large_building.py \
  -i large_scan.ply \
  -o clean_large.ply
```

Accepts the same core arguments as `preprocess_for_spatiallm.py`. Additional logic handles
point clouds that exceed a configurable point count by processing them in spatial tiles.

---

## 5. inference.py

Runs SpatialLM inference: loads the point cloud, encodes it with the Sonata point cloud
encoder, and generates a structured layout using the language model head.

**Usage**
```bash
python inference.py \
  --point_cloud clean_scan.ply \
  --output layout.txt \
  --model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
  --detect_type all
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `-p` / `--point_cloud` | required | Input `.ply` file or folder of `.ply` files |
| `-o` / `--output` | required | Output layout `.txt` file or folder |
| `-m` / `--model_path` | `SpatialLM-Llama-1B` | HuggingFace model name or local path |
| `-d` / `--detect_type` | `all` | `all` / `arch` (walls+openings) / `object` (furniture) |
| `-c` / `--category` | all 59 | Subset of furniture categories (see full list below) |
| `--repetition_penalty` | `1.1` | Penalises repeated tokens; values >1.0 reduce loops |
| `--inference_dtype` | `bfloat16` | Use `float32` on GPUs that do not support bfloat16 |
| `--no_cleanup` | off | Skip internal point cloud denoising |
| `--seed` | `-1` | Set a non-negative integer for reproducible results |

**Decoding behaviour**

Inference uses greedy decoding (`do_sample=False`, `num_beams=1`). The `--top_k`,
`--top_p`, and `--temperature` flags are accepted for CLI compatibility but do not affect
output when `do_sample=False`. For reproducibility, a global seed is set at module load
time (SEED = 42) in addition to the per-call `--seed` argument.

**Supported furniture categories**

```
sofa  chair  dining_chair  bar_chair  stool  bed  pillow  wardrobe  nightstand
tv_cabinet  wine_cabinet  bathroom_cabinet  shoe_cabinet  entrance_cabinet
decorative_cabinet  washing_cabinet  wall_cabinet  sideboard  cupboard
coffee_table  dining_table  side_table  dressing_table  desk  integrated_stove
gas_stove  range_hood  micro-wave_oven  sink  stove  refrigerator  hand_sink
shower  shower_room  toilet  tub  illumination  chandelier  floor-standing_lamp
wall_decoration  painting  curtain  carpet  plants  potted_bonsai  tv  computer
air_conditioner  washing_machine  clothes_rack  mirror  bookcase  cushion  bar
screen  combination_sofa  dining_table_combination
leisure_table_and_chair_combination  multifunctional_combination_bed
```

---

## 6. postprocess.py

Corrects common SpatialLM detection errors using geometry before IFC generation.

**Usage**
```bash
python postprocess.py \
  --layout raw_layout.txt \
  --point_cloud clean_scan.ply \
  --output refined_layout.txt
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `-l` / `--layout` | required | Raw SpatialLM layout TXT |
| `-p` / `--point_cloud` | required | Cleaned point cloud `.ply` |
| `-o` / `--output` | required | Corrected layout TXT |
| `--angle_thresh` | `10.0°` | Max angle deviation for two walls to be considered collinear |
| `--dist_thresh` | `0.35 m` | Max perpendicular distance between wall midpoints for merging |
| `--gap_thresh` | `0.60 m` | Max projection gap between two wall segments for merging |
| `--min_points` | `50` | Min point cloud points inside a bbox to trigger refitting |

### Algorithm 1 — Collinear wall detection

SpatialLM sometimes splits a single real wall into two segments. Three geometric tests must
all pass before two walls are considered collinear:

```
are_collinear(w1, w2):

  Test 1 — Angle (mod 180° to ignore direction)
    a1 = atan2(dy1, dx1) mod 180°
    a2 = atan2(dy2, dx2) mod 180°
    diff = min(|a1−a2|, 180−|a1−a2|)
    → diff < angle_thresh

  Test 2 — Perpendicular distance
    midpoint of w2: (mx, my)
    d = |−dy1·(mx−x1) + dx1·(my−y1)| / ‖(dx1,dy1)‖
    → d < dist_thresh

  Test 3 — Projection gap
    Project all 4 endpoints onto the unit axis of w1
    gap = max(0, right_segment_start − left_segment_end)
    → gap < gap_thresh
```

### Algorithm 2 — Wall merging

When collinearity is confirmed, the two walls are replaced by one spanning the full range:

```
merge_walls(w1, w2):
  Project all 4 endpoints onto the unit axis of w1
  t_min → new_start = origin_w1 + t_min × unit(w1)
  t_max → new_end   = origin_w1 + t_max × unit(w1)
  height    = max(h1, h2)
  thickness = max(t1, t2)  or 0.2 if both are 0
```

Merging is **iterative** — repeated until no new collinear pairs are found.

### Algorithm 3 — Points-inside-bbox test (Delaunay)

To find the point cloud points actually inside an oriented bounding box:

```
1. Compute the 8 corners of the bbox in world space:
     local corners (±l/2, ±w/2, ±h/2) rotated by bbox.rotation around Z
     then translated to (cx, cy, cz)
2. hull = Delaunay(8 corners)
3. mask = hull.find_simplex(points) >= 0
4. return points[mask]
```

Delaunay is used as a convex polytope inclusion test, which correctly handles the bbox
rotation unlike an axis-aligned bounding box check.

### Algorithm 4 — Minimum bounding rectangle (rotating calipers)

To refit a bbox to the actual point cloud points it contains:

```
fit_min_bbox_2d(points):
  1. Project points to 2D (drop Z)
  2. Compute ConvexHull — N edges
  3. For each edge:
       a. Define a local frame (axis = edge direction)
       b. Project all hull points into this frame
       c. Compute axis-aligned bounding rectangle
       d. Record area = width × length
  4. Keep the edge that minimises area
  → optimal angle, new (cx, cy), length, width
  Z and height are kept from the original SpatialLM prediction
```

Complexity O(n log n) dominated by the convex hull step. This is the standard
**minimum area bounding rectangle** algorithm.

---

## 7. spatiallm_to_ifc.py

Converts a layout TXT file into an IFC4 building model.

**Usage**
```bash
python spatiallm_to_ifc.py \
  --input refined_layout.txt \
  --output scene.ifc
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `-i` / `--input` | required | Layout TXT (raw or refined) |
| `-o` / `--output` | `output.ifc` | Output IFC file |

### IFC hierarchy

```
IfcProject
  └─ IfcSite
       └─ IfcBuilding
            └─ IfcBuildingStorey  ("Ground Floor")
                 ├─ IfcSlab              ← floor slab (convex hull of wall endpoints)
                 ├─ IfcWall × N
                 ├─ IfcDoor × N
                 ├─ IfcWindow × N
                 └─ IfcFurnishingElement × N
```

### Geometry — IfcExtrudedAreaSolid

All solids are built by the same method: `IfcRectangleProfileDef` extruded vertically with
`IfcExtrudedAreaSolid`. Dimensions used for each element type:

| Element | Width (profile X) | Depth (profile Y) | Height (extrusion) |
|---|---|---|---|
| Wall | `‖(x2,y2)−(x1,y1)‖` | `thickness` (0.2 m if 0) | `height` |
| Door | `width` | parent wall thickness | `height` |
| Window | `width` | parent wall thickness | `height` |
| Bbox | `length` | `width` | `height` |
| Slab | convex hull polygon | — | 0.10 m |

### Geometry — IfcAxis2Placement3D

Each element is positioned by a local frame (origin, Z-axis, X-axis):

| Element | Origin | X-axis |
|---|---|---|
| Wall | `(x1, y1, z1)` | `(cos θ, sin θ, 0)` where `θ = atan2(y2−y1, x2−x1)` |
| Door | `(cx, cy, cz − height/2)` | direction of parent wall |
| Window | `(cx, cy, cz − height/2)` | direction of parent wall |
| Bbox | `(cx, cy, cz − height/2)` | `(cos r, sin r, 0)` where `r` = rotation |
| Slab | `(0, 0, z_min − 0.10)` | `(1, 0, 0)` |

The Y-axis is derived implicitly as the cross product Z × X.

### Geometry — Floor slab

```
1. Collect all wall endpoint XY pairs: [(x1,y1), (x2,y2)] for each wall
2. Compute 2D ConvexHull
3. Build IfcPolyline from hull vertices (closed)
4. IfcArbitraryClosedProfileDef(polyline)
5. Extrude 10 cm downward from z_min − 0.10 m
```

### Wall direction and thickness lookups

Before processing doors and windows, two dictionaries are built from the wall list:

```python
wall_dir_map[w.name]       = (dx / length, dy / length)  # unit vector
wall_thickness_map[w.name] = w.thickness if w.thickness > 0.001 else 0.2
```

Each door and window references `parent_wall` to retrieve the correct orientation and
thickness for its host wall.

---

## 8. run_pipeline.py

Orchestrates all four pipeline steps in a single command with timing output.

**Usage**
```bash
# Minimal
python run_pipeline.py --input scan.ply --output model.ifc

# Full
python run_pipeline.py \
  --input  scan.ply \
  --output model.ifc \
  --workdir /tmp/pipeline \
  --model  manycore-research/SpatialLM1.1-Qwen-0.5B \
  --detect_type all \
  --seed 42 \
  --preprocess_mode moderate \
  --target_height 2.5 \
  --skip_preprocess \
  --skip_postprocess \
  --inference_python /opt/conda/envs/spatiallm/bin/python
```

**All arguments**

| Argument | Default | Description |
|---|---|---|
| `--input` / `-i` | required | Raw `.ply` point cloud |
| `--output` / `-o` | `<stem>.ifc` | Output IFC file |
| `--workdir` / `-w` | input directory | Folder for intermediate files |
| `--model` / `-m` | `SpatialLM-Llama-1B` | Model name or local path |
| `--detect_type` | `all` | `all` / `arch` / `object` |
| `--seed` | `42` | Inference seed for reproducibility |
| `--preprocess_mode` | `moderate` | Denoising intensity |
| `--target_height` | `2.5` | Expected ceiling height in metres |
| `--skip_preprocess` | off | Use raw input directly |
| `--skip_postprocess` | off | Skip geometric correction |
| `--inference_python` | `sys.executable` | Alternate Python interpreter for inference step |
| `--angle_thresh` | `10.0°` | Wall merge angle threshold |
| `--dist_thresh` | `0.35 m` | Wall merge distance threshold |
| `--gap_thresh` | `0.60 m` | Wall merge gap threshold |
| `--min_points` | `50` | Min points for bbox refitting |

**Intermediate files produced**

```
<workdir>/
  <stem>_clean.ply           ← cleaned point cloud
  <stem>_raw_layout.txt      ← raw SpatialLM output
  <stem>_refined_layout.txt  ← geometrically corrected layout
<output>.ifc                 ← final BIM file
```

**Using a separate conda environment for inference**

If SpatialLM dependencies (torch, transformers) live in a different environment:

```bash
python run_pipeline.py --input scan.ply \
  --inference_python /opt/conda/envs/spatiallm/bin/python
```

---

## 9. pipeline_gui.py

Browser-based Streamlit interface covering the full pipeline.

**Launch**
```bash
streamlit run pipeline_gui.py
# Open http://localhost:8501
```

**Features**

- Drag-and-drop `.ply` upload
- Sidebar controls for every pipeline parameter
- Live log streaming for each script during execution
- Side-by-side comparison of raw vs refined layout (element counts and deltas)
- Interactive 3D maquette (Plotly): floor polygon with Delaunay triangulation, filled walls,
  door/window panels, semi-transparent furniture boxes with floating category labels
- Embedded Rerun viewer (launched on demand) for the raw point cloud + detected layout
- Download buttons: cleaned PLY · raw layout TXT · refined layout TXT · IFC file

**Sidebar sections**

| Section | Controls |
|---|---|
| Preprocessing | Mode (full / alignment-only / skip), denoising, largest cluster, colours, target height |
| Inference | Model (Llama 1B / Qwen 0.5B), detection type, seed |
| Post-processing | Enable/skip, angle threshold, distance threshold, gap threshold, min points |
| IFC | Enable/skip |

---

## 10. Docker

The `Dockerfile` and `docker-compose.yml` package the entire stack into a portable GPU image.

**Build**
```bash
docker compose build
```

The first build takes 20–40 minutes because flash-attn compiles from source. Subsequent
builds are fast — the poetry dependency layer is cached and only re-runs when
`pyproject.toml` or `poetry.lock` changes.

**Run — Streamlit GUI**
```bash
docker compose up
# Open http://localhost:8501
```

**Run — CLI pipeline**
```bash
docker compose run spatiallm python run_pipeline.py \
  --input /app/data/scan.ply \
  --output /app/output/model.ifc \
  --model manycore-research/SpatialLM1.1-Qwen-0.5B
```

**Volume mounts**

| Mount | Purpose |
|---|---|
| `./data:/app/data` | Input `.ply` files (bind mount from host) |
| `./output:/app/output` | Generated IFC and intermediate files |
| `hf_cache:/root/.cache/huggingface` | HuggingFace model cache (named volume, persists across runs) |

Model weights are downloaded on first run and cached in the named volume — they are not
baked into the image.

**Requirements on the host**

- Docker Engine
- NVIDIA driver
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## 11. IFC output structure

The generated `.ifc` file uses the IFC4 schema and can be opened in any conformant BIM
viewer: BlenderBIM, FreeCAD, ArchiCAD, Revit, Solibri, etc.

**Hierarchy:**
```
IfcProject  "Indoor3D Project"
  IfcSite  "Site"
    IfcBuilding  "Building"
      IfcBuildingStorey  "Ground Floor"
        IfcSlab          (floor slab — convex hull footprint, 10 cm thick)
        IfcWall × N      (one per wall segment)
        IfcDoor × N      (one per detected door)
        IfcWindow × N    (one per detected window)
        IfcFurnishingElement × N  (one per Bbox — labelled with furniture category)
```

All elements are placed using `IfcLocalPlacement` relative to the building storey origin.
Geometry is `IfcExtrudedAreaSolid` throughout. Each element carries a unique GUID generated
at conversion time.

---

## 12. Known limitations

**Doors and windows are solids, not openings.**
IFC correctly distinguishes openings (`IfcOpeningElement`) from the elements that fill them
(`IfcRelFillsElement`). This pipeline places door and window solids inside walls but does
not cut holes. Proper openings require wall geometry subtraction, which SpatialLM does not
provide. Structural analysis tools that rely on wall continuity will not see correct results.

**Wall thickness defaults to 0.2 m when SpatialLM predicts 0.0.**
SpatialLM does not always predict wall thickness reliably. The fallback (0.2 m) is a
reasonable approximation for residential construction but may differ from reality.

**Single floor / single room.**
`run_pipeline.py` processes one `.ply` at a time and generates a single `IfcBuildingStorey`.
Multi-room or multi-floor merging is not yet implemented.

**Flash Attention is disabled on pre-Ampere GPUs (CC < 8.0).**
The Sonata encoder detects GPU compute capability at startup and falls back to standard
attention automatically. Inference is slightly slower on these devices but functionally
identical.

**Inference is deterministic but model-quality-limited.**
With `do_sample=False` and a fixed seed, results are fully reproducible. Layout quality
depends entirely on SpatialLM's predictions — post-processing corrects common geometric
errors but cannot recover missing walls or misdetected elements.
