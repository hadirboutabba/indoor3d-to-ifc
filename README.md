# Indoor3D-to-IFC

A pipeline for converting 3D point cloud scans of residential buildings into structured **IFC (Industry Foundation Classes)** BIM models, using [SpatialLM](https://github.com/manycore-research/SpatialLM) as the scene-understanding backbone.

---

## Overview

This project automates the journey from a raw 3D scan of a residential interior to a standards-compliant IFC building model:

```
Slamtec Aurora S (or any scanner)
        │
        ▼ PLY point cloud
  ┌─────────────┐
  │  SpatialLM  │  ← detects walls, doors, windows, furniture (oriented bounding boxes)
  └─────────────┘
        │
        ▼ structured layout (.txt)
  ┌─────────────┐
  │ IFC Builder │  ← converts spatial entities to IFC elements (IfcWall, IfcDoor, …)
  └─────────────┘
        │
        ▼ .ifc model
  Any BIM / CAD tool (Revit, ArchiCAD, BlenderBIM, …)
```

**Supported input sources**

| Scanner | Output format | Notes |
|---|---|---|
| Slamtec Aurora S | `.ply` | Primary target — exports axis-aligned point clouds |
| Any LiDAR sensor | `.ply` | Z-axis must be the up axis |
| Monocular video + MASt3R-SLAM | `.ply` | Works well for quick surveys |
| RGBD cameras | `.ply` | After point cloud reconstruction |

**What SpatialLM detects**

| Entity | IFC equivalent |
|---|---|
| Wall segment (start/end/height/thickness) | `IfcWall` |
| Door (wall reference, position, size) | `IfcDoor` |
| Window (wall reference, position, size) | `IfcWindow` |
| Oriented bounding box (59 furniture categories) | `IfcFurnishingElement` |

---

## Models

SpatialLM weights used by this project:

| Model | Size | Link |
|---|---|---|
| SpatialLM1.1-Qwen-0.5B | 0.5B | [HuggingFace](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B) |
| SpatialLM1.1-Llama-1B | 1B | [HuggingFace](https://huggingface.co/manycore-research/SpatialLM1.1-Llama-1B) |

---

## Installation

**Requirements:** Python 3.11 · PyTorch 2.4.1 · CUDA 12.4

```bash
git clone https://github.com/YOUR_USERNAME/indoor3d-to-ifc.git
cd indoor3d-to-ifc

conda create -n indoor3d python=3.11
conda activate indoor3d
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit conda-forge::sparsehash

pip install poetry && poetry config virtualenvs.create false --local
poetry install

# Install Sonata point cloud encoder (required for SpatialLM 1.1)
poe install-sonata
```

> **Note on Flash Attention:** If your GPU does not support Flash Attention or you encounter
> installation issues, the encoder automatically falls back to standard attention.

---

## Usage

### Step 1 — Prepare your point cloud

Export your scan as a `.ply` file with the **Z-axis pointing up**. Most Slamtec Aurora S exports are already axis-aligned. If not, align them before running inference.

### Step 2 — Run SpatialLM inference

```bash
# Full structured reconstruction (walls + doors + windows + furniture)
python inference.py \
  --point_cloud path/to/scan.ply \
  --output path/to/output.txt \
  --model_path manycore-research/SpatialLM1.1-Qwen-0.5B

# Layout only (walls, doors, windows — no furniture)
python inference.py \
  --point_cloud path/to/scan.ply \
  --output path/to/output.txt \
  --model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
  --detect_type arch

# Specific furniture categories only
python inference.py \
  --point_cloud path/to/scan.ply \
  --output path/to/output.txt \
  --model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
  --detect_type object \
  --category sofa bed dining_table
```

**Key inference options**

| Flag | Default | Description |
|---|---|---|
| `--detect_type` | `all` | `all` / `arch` / `object` |
| `--category` | (all 59) | Subset of furniture categories to detect |
| `--repetition_penalty` | `1.1` | Reduces repetitive predictions |
| `--inference_dtype` | `bfloat16` | Use `float32` if bfloat16 is unsupported |
| `--no_cleanup` | off | Skip point cloud denoising |
| `--seed` | -1 | Set for reproducible results |

### Step 3 — Visualize (optional)

```bash
python visualize.py \
  --point_cloud path/to/scan.ply \
  --layout path/to/output.txt \
  --save preview.rrd

rerun preview.rrd
```

### Step 4 — Convert to IFC

> **Work in progress.** The IFC builder module is under active development.
> See [PIPELINE.md](./PIPELINE.md) for the planned architecture.

```bash
# Coming soon
python ifc_builder.py --layout path/to/output.txt --output model.ifc
```

---

## Output format

The SpatialLM layout text file uses a Python dataclass-style schema (discretized integers):

```python
wall_0 = Wall(ax=..., ay=..., az=..., bx=..., by=..., bz=..., height=..., thickness=...)
door_0 = Door(wall_id='wall_0', position_x=..., position_y=..., position_z=..., width=..., height=...)
window_0 = Window(wall_id='wall_0', position_x=..., position_y=..., position_z=..., width=..., height=...)
bbox_0 = Bbox(class='sofa', position_x=..., position_y=..., position_z=..., angle_z=..., scale_x=..., scale_y=..., scale_z=...)
```

See [PIPELINE.md](./PIPELINE.md) for how these map to IFC entities.

---

## Development environment

The pipeline is currently developed and tested on the following setup:

**Host machine**

| Component | Details |
|---|---|
| OS | Windows 11 + WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| CPU | Intel Core i7-10875H @ 2.30 GHz (8 cores) |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 2070 Super (8 GB VRAM, compute capability 7.5) |
| GPU driver | 581.95 |

**Software environment (inside WSL2)**

| Component | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.4.1+cu124 |
| CUDA toolkit | 12.4 |
| Conda environment | `indoor3d` |

**Hardware notes**

- The RTX 2070 Super (Turing, CC 7.5) does **not** support Flash Attention, which requires Ampere (CC 8.0+) or newer. Flash Attention is therefore disabled in `spatiallm/model/sonata_encoder.py` and standard attention is used instead.
- The 0.5B Qwen model fits comfortably in 8 GB VRAM during inference. The 1B Llama model requires approximately 6–7 GB and also runs on this card.
- Full fine-tuning (~60 GB VRAM requirement) is not feasible on this machine. The project targets inference and lightweight adaptation only.

---

## Project structure

```
.
├── inference.py              # SpatialLM inference — point cloud → layout text
├── visualize.py              # Rerun-based 3D preview
├── eval.py                   # Benchmark evaluation
├── train.py                  # Fine-tuning entry point
├── code_template.txt         # Schema fed to the LLM as generation context
├── configs/
│   └── spatiallm_sft.yaml    # Fine-tuning config
├── spatiallm/
│   ├── model/                # SpatialLM model definitions (Llama / Qwen variants + Sonata encoder)
│   ├── layout/               # Layout parsing and coordinate handling
│   ├── pcd/                  # Point cloud loading and preprocessing
│   └── tuner/                # Fine-tuning pipeline (data, trainer, hyperparams)
└── PIPELINE.md               # Detailed pipeline design and IFC mapping
```

---

## Roadmap

- [x] SpatialLM inference on PLY point clouds
- [x] Repetition penalty + attention mask fixes for stable generation
- [ ] IFC builder: Wall → `IfcWall`
- [ ] IFC builder: Door/Window → `IfcDoor` / `IfcWindow`
- [ ] IFC builder: Furniture OBBs → `IfcFurnishingElement`
- [ ] Slamtec Aurora S pre-processing script (coordinate alignment)
- [ ] Multi-room / multi-floor merge
- [ ] Evaluation on residential scan dataset

---

## Credits

- **SpatialLM** — [manycore-research/SpatialLM](https://github.com/manycore-research/SpatialLM) (NeurIPS 2025)
- **Sonata encoder** — [xywu.me/sonata](https://xywu.me/sonata/)
- **MASt3R-SLAM** — for monocular video reconstruction

---

## License

The SpatialLM model weights are subject to the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/).
Code in this repository is MIT licensed unless otherwise noted.
