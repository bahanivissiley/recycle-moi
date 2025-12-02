# ============================================
# STAGE 1: Builder
# ============================================
FROM python:3.11-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Créer environnement virtuel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copier et installer les dépendances
COPY requirements-docker.txt .

# Installer PyTorch CPU-only (plus léger pour Docker)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Installer le reste
RUN pip install --no-cache-dir -r requirements-docker.txt

# ============================================
# STAGE 2: Runtime
# ============================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Installer dépendances runtime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier environnement virtuel
COPY --from=builder /opt/venv /opt/venv

# Créer utilisateur non-root
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copier le code
COPY --chown=appuser:appuser . .

USER appuser

# Exposer le port (configurable via variable d'environnement)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Commande de démarrage avec port configurable
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}