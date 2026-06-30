# Indoor3D-to-IFC

Convert a 3D point cloud scan of a building into a standards-compliant **IFC (Industry Foundation Classes)** BIM model using [SpatialLM](https://github.com/manycore-research/SpatialLM) as the scene-understanding backbone.

```
raw scan (.ply)
      │
      ▼  preprocess_for_spatiallm.py
clean, axis-aligned cloud
      │
      ▼  inference.py  (SpatialLM 1.1)
structured layout (.txt)
      │
      ▼  postprocess.py
corrected layout (.txt)
      │
      ▼  spatiallm_to_ifc.py
BIM model (.ifc)
```

---

## What it detects

| SpatialLM output | IFC element |
|---|---|
| Wall segment (start · end · height · thickness) | `IfcWall` |
| Door (parent wall · position · size) | `IfcDoor` |
| Window (parent wall · position · size) | `IfcWindow` |
| Oriented bounding box (59 furniture categories) | `IfcFurnishingElement` |
| Room footprint (convex hull of wall endpoints) | `IfcSlab` |

---

## Models

| Model | Size | HuggingFace |
|---|---|---|
| SpatialLM1.1-Llama-1B | 1B | [manycore-research/SpatialLM1.1-Llama-1B](https://huggingface.co/manycore-research/SpatialLM1.1-Llama-1B) |
| SpatialLM1.1-Qwen-0.5B | 0.5B | [manycore-research/SpatialLM1.1-Qwen-0.5B](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B) |

Model weights are downloaded from HuggingFace automatically on first run.

---

## Quick start — Docker (recommended)

Docker packages the entire CUDA + Sonata stack. No conda environment to manage on the target device.

**Requirements:** Docker, NVIDIA driver, [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
git clone https://github.com/hadirboutabba/indoor3d-to-ifc.git
cd indoor3d-to-ifc

# Build the image (~20–40 min on first run — flash-attn compiles from source)
docker compose build

# Launch the Streamlit GUI at http://localhost:8501
docker compose up
```

Place your `.ply` files in `./data/` — they are bind-mounted into the container at `/app/data/`.
Outputs are written to `./output/`. Model weights are cached in a persistent Docker volume and
downloaded only once.

**CLI pipeline via Docker:**
```bash
docker compose run spatiallm python run_pipeline.py \
  --input /app/data/scan.ply \
  --output /app/output/model.ifc \
  --model manycore-research/SpatialLM1.1-Qwen-0.5B
```

---

## Installation — native (WSL2 / Linux)

**Requirements:** Python 3.11 · PyTorch 2.4.1 · CUDA 12.4

```bash
git clone https://github.com/hadirboutabba/indoor3d-to-ifc.git
cd indoor3d-to-ifc

conda create -n indoor3d python=3.11
conda activate indoor3d
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit conda-forge::sparsehash

pip install poetry && poetry config virtualenvs.create false --local
poetry install

# Install Sonata point cloud encoder (required for SpatialLM 1.1)
poe install-sonata
```

> Flash Attention requires Ampere (CC 8.0+) or newer. On older GPUs (e.g. Turing CC 7.5) the
> encoder automatically falls back to standard attention — no action needed.

---

## Usage

### Option A — Streamlit GUI

```bash
streamlit run pipeline_gui.py
# Open http://localhost:8501
```

The GUI covers the full pipeline from file upload to IFC download:

- Upload a `.ply` by drag-and-drop
- Configure each step from the sidebar (model, denoising mode, detection type, post-processing thresholds)
- Live log streaming during execution
- Interactive 3D maquette (Plotly) showing walls, doors, windows, and furniture with labels
- Rerun point cloud viewer (launched on demand)
- Download buttons for every intermediate and final output

### Option B — One-command CLI pipeline

```bash
# Minimal
python run_pipeline.py --input scan.ply --output model.ifc

# Full options
python run_pipeline.py \
  --input  scan.ply \
  --output model.ifc \
  --model  manycore-research/SpatialLM1.1-Llama-1B \
  --detect_type all \
  --preprocess_mode moderate \
  --seed 42 \
  --workdir /tmp/pipeline_work
```

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | required | Raw `.ply` point cloud |
| `--output` / `-o` | `<stem>.ifc` | Output IFC file |
| `--workdir` / `-w` | input directory | Folder for intermediate files |
| `--model` / `-m` | `SpatialLM1.1-Llama-1B` | HuggingFace model name or local path |
| `--detect_type` | `all` | `all` / `arch` (walls+openings) / `object` (furniture only) |
| `--seed` | `42` | Random seed for reproducible inference |
| `--preprocess_mode` | `moderate` | `conservative` / `moderate` / `aggressive` |
| `--target_height` | `2.5` | Expected ceiling height in metres |
| `--skip_preprocess` | off | Use the raw PLY as-is |
| `--skip_postprocess` | off | Skip geometric correction |
| `--inference_python` | current python | Alternate interpreter (useful when SpatialLM lives in a separate conda env) |

### Option C — Step by step

**Step 1 — Preprocess**
```bash
python preprocess_for_spatiallm.py \
  -i raw_scan.ply -o clean_scan.ply \
  --mode moderate --target_height 2.5
```

**Step 2 — Inference**
```bash
python inference.py \
  --point_cloud clean_scan.ply \
  --output layout_raw.txt \
  --model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
  --detect_type all
```

**Step 3 — Post-process**
```bash
python postprocess.py \
  --layout layout_raw.txt \
  --point_cloud clean_scan.ply \
  --output layout_refined.txt
```

**Step 4 — Generate IFC**
```bash
python spatiallm_to_ifc.py \
  --input layout_refined.txt \
  --output model.ifc
```

---

## Supported input sources

| Source | Format | Notes |
|---|---|---|
| Slamtec Aurora S | `.ply` | Primary target |
| Any LiDAR sensor | `.ply` | Z-axis must be up |
| Monocular video + MASt3R-SLAM | `.ply` | Good for quick surveys |
| RGBD cameras | `.ply` | After point cloud reconstruction |

---

## Project structure

```
indoor3d-to-ifc/
├── preprocess_for_spatiallm.py  # Denoise · Z-up align · Manhattan align · metric scale
├── preprocess_large_building.py # Variant for large / multi-room buildings
├── inference.py                 # SpatialLM inference: point cloud → layout TXT
├── postprocess.py               # Geometric correction: merge collinear walls, refit bboxes
├── spatiallm_to_ifc.py          # IFC generation: layout TXT → .ifc
├── run_pipeline.py              # CLI orchestrator: runs all four steps in sequence
├── pipeline_gui.py              # Streamlit GUI: full pipeline in a browser tab
├── spatiallm_gui.py             # Original 3-step GUI (preprocess → infer → visualize)
├── visualize.py                 # Rerun-based 3D preview
├── eval.py                      # Benchmark evaluation
├── train.py                     # Fine-tuning entry point
├── code_template.txt            # Schema fed to SpatialLM as generation context
├── Dockerfile                   # CUDA 12.4 + full stack image
├── docker-compose.yml           # GPU passthrough, volume mounts, port mapping
├── configs/
│   └── spatiallm_sft.yaml       # Fine-tuning configuration
└── spatiallm/
    ├── model/                   # SpatialLM model (Llama / Qwen + Sonata encoder)
    ├── layout/                  # Layout parsing and coordinate handling
    ├── pcd/                     # Point cloud loading and preprocessing
    └── tuner/                   # Fine-tuning pipeline
```

---

## Hardware requirements

| Component | Minimum | Notes |
|---|---|---|
| GPU VRAM | 6 GB | Qwen 0.5B: ~4 GB · Llama 1B: ~6–7 GB |
| CUDA | 12.4 | Matches PyTorch 2.4.1+cu124 |
| Flash Attention | optional | Requires Ampere (CC 8.0+); falls back automatically on older GPUs |
| RAM | 16 GB | For point cloud loading |

---

## Roadmap

- [x] SpatialLM inference on PLY point clouds
- [x] Greedy / deterministic decoding (fixed seed, `do_sample=False`)
- [x] Preprocessing — denoise, Z-up + Manhattan alignment, metric scaling
- [x] Geometric post-processing — collinear wall merging, bbox refitting
- [x] IFC generation — `IfcWall`, `IfcDoor`, `IfcWindow`, `IfcFurnishingElement`, `IfcSlab`
- [x] CLI pipeline orchestrator (`run_pipeline.py`)
- [x] Streamlit GUI with live logs, 3D maquette, and download buttons
- [x] Docker image for portable GPU deployment
- [ ] True IFC openings (`IfcOpeningElement`) for doors and windows
- [ ] Multi-room / multi-floor merging
- [ ] Evaluation on a residential scan dataset

---

## Credits

- **SpatialLM** — [manycore-research/SpatialLM](https://github.com/manycore-research/SpatialLM) (NeurIPS 2025)
- **Sonata encoder** — [xywu.me/sonata](https://xywu.me/sonata/)
- **MASt3R-SLAM** — for monocular video-to-point-cloud reconstruction

---

## License

SpatialLM model weights are subject to the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/).
Code in this repository is MIT licensed unless otherwise noted.
