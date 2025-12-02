"""
Application FastAPI principale pour Recycle-moi
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import health, model, predict
from api.utils.model_loader import model_loader

# Chemin vers le modèle
MODEL_PATH = project_root / "checkpoints" / "v1.0" / "best_model.pth"
METADATA_PATH = project_root / "checkpoints" / "v1.0" / "metadata.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application
    Charge le modèle au démarrage
    """
    # Startup
    print("🚀 Démarrage de l'API Recycle-moi...")
    
    # Charger le modèle
    try:
        model_loader.load(
            checkpoint_path=str(MODEL_PATH),
            metadata_path=str(METADATA_PATH)
        )
        print("✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        raise
    
    yield
    
    # Shutdown
    print("🛑 Arrêt de l'API...")

# Créer l'application FastAPI
app = FastAPI(
    title="Recycle-moi API",
    description="API de classification de déchets par deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS (pour permettre les requêtes depuis le frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routes
app.include_router(health.router, tags=["Health"])
app.include_router(model.router, tags=["Model"])
app.include_router(predict.router, tags=["Prediction"])

@app.get("/")
async def root():
    """
    Endpoint racine
    """
    return {
        "message": "Bienvenue sur l'API Recycle-moi",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Lancer le serveur
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload en développement
    )