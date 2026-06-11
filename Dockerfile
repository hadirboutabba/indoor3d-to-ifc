# ── Base: CUDA 12.4 + cuDNN (matches pyproject.toml torch cu124 requirement) ──
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl build-essential ninja-build \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda (needed for sparsehash from conda-forge) ────────────────────────
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# ── Python 3.11 + sparsehash in conda base env ────────────────────────────────
RUN conda install -y python=3.11 \
        -c nvidia/label/cuda-12.4.0 \
        -c conda-forge \
        cuda-toolkit sparsehash \
    && conda clean -afy

# ── Poetry ────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

WORKDIR /app

# ── Install Python deps via Poetry (cache layer — copy lock files first) ──────
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# ── Sonata extras: flash-attn (long compile), torch-scatter, spconv ────────────
RUN pip install --no-cache-dir ninja flash-attn --no-build-isolation
RUN pip install --no-cache-dir torch-scatter \
        -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
RUN pip install --no-cache-dir timm spconv-cu120

# ── Copy the rest of the project ──────────────────────────────────────────────
COPY . .

# ── Streamlit port ────────────────────────────────────────────────────────────
EXPOSE 8501

# Default: launch the Streamlit GUI
# Override with `docker run ... python run_pipeline.py ...` for CLI use
CMD ["streamlit", "run", "pipeline_gui.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
