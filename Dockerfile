# ── Base légère : les wheels torch cu124 embarquent CUDA + cuDNN ──────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.5" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── Système + Python 3.11 en une seule couche ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg software-properties-common \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && pip install --no-cache-dir poetry \
    && apt-get purge -y software-properties-common gnupg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache

WORKDIR /app

ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_CACHE=true \
    PIP_DEFAULT_TIMEOUT=600 \
    POETRY_REQUESTS_TIMEOUT=600

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-cache && rm -rf /root/.cache

ENV PATH="/app/.venv/bin:$PATH"

# ── Deps hors pyproject (Sonata) ──────────────────────────────────────────────
# torch-scatter est compilé depuis le sdist PyPI avec --no-build-isolation :
# l'index data.pyg.org n'est pas résolvable ici, et sans cette option
# l'environnement de build isolé de pip ne trouve pas torch.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir timm spconv-cu120 \
    && pip install --no-cache-dir torch-scatter --no-build-isolation \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && find /app/.venv -name "*.a" -delete \
    && find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# streamlit et ifcopenshell : déjà dans pyproject.toml
# flash-attn : supprimé (inutilisable sur CC 7.5, fallback géré par flash_attn_utils.py)

COPY . .

EXPOSE 8501
CMD ["/app/.venv/bin/streamlit", "run", "pipeline_gui.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]