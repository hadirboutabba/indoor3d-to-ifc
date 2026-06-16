# ── Base: CUDA 12.4 + cuDNN (matches pyproject.toml torch cu124 requirement) ──
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.5"

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl build-essential ninja-build software-properties-common \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
        libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.11 (Ubuntu 22.04 ships 3.10 by default) ──────────────────────────
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3.11 the default `python` / `python3` ──────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ── Poetry (installed via system pip, isolated from project deps) ─────────────
RUN pip install --no-cache-dir --break-system-packages poetry

WORKDIR /app

# ── Let Poetry manage its own virtualenv for project deps (avoids system   ────
# ── dist-packages conflicts like the apt-managed 'blinker' distutils pkg)  ────
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    PIP_DEFAULT_TIMEOUT=600 \
    POETRY_REQUESTS_TIMEOUT=600
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# ── Put the poetry-managed venv first on PATH for subsequent pip installs ─────
ENV PATH="/app/.venv/bin:$PATH"

# ── Streamlit (not currently in pyproject.toml/poetry.lock) ───────────────────
RUN pip install --no-cache-dir streamlit

# ── Sonata extras: torch-scatter, spconv (required) ────────────────────────────
RUN pip install --no-cache-dir torch-scatter \
        -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
RUN pip install --no-cache-dir timm spconv-cu120

# ── Flash Attention: optional, requires CC 8.0+ (Ampere or newer) ─────────────
# On Turing GPUs (CC 7.5 — RTX 2070 Super / 2080 Ti) this either fails to build
# or produces a binary that can't be used at runtime. The Sonata encoder falls
# back to standard attention automatically when flash-attn is absent, so we
# allow this step to fail without breaking the image build.
RUN pip install --no-cache-dir ninja flash-attn --no-build-isolation || \
    echo "flash-attn install skipped/failed (expected on CC < 8.0) — fallback attention will be used"

# ── Copy the rest of the project ──────────────────────────────────────────────
COPY . .

# ── Streamlit port ────────────────────────────────────────────────────────────
EXPOSE 8501

# Default: launch the Streamlit GUI
# Override with `docker run ... python run_pipeline.py ...` for CLI use
CMD ["/app/.venv/bin/streamlit", "run", "pipeline_gui.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]