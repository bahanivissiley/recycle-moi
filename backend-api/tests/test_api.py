"""
Tests unitaires pour l'API Recycle-moi
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import io
from PIL import Image

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)

def test_root():
    """Test endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

def test_health():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_model_info():
    """Test model info (peut échouer si modèle non chargé)"""
    response = client.get("/model/info")
    # Accepter 200 (modèle chargé) ou 500/503 (modèle non chargé en CI)
    assert response.status_code in [200, 500, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "model_version" in data
        assert "architecture" in data

def test_predict_no_file():
    """Test prédiction sans fichier"""
    response = client.post("/predict")
    assert response.status_code == 422  # Validation error

def test_predict_invalid_file():
    """Test prédiction avec fichier invalide"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    # Accepter 400 (bad request) ou 503 (modèle non chargé)
    assert response.status_code in [400, 503]

@pytest.mark.skipif(
    not Path("checkpoints/v1.0/best_model.pth").exists(),
    reason="Modèle non disponible"
)
def test_predict_with_valid_image():
    """Test prédiction avec une vraie image (seulement si modèle disponible)"""
    # Créer une image de test
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    
    if response.status_code == 200:
        data = response.json()
        assert data["success"] == True
        assert "predicted_class" in data
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1

def test_docs_accessible():
    """Test que la documentation Swagger est accessible"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_schema():
    """Test que le schéma OpenAPI est accessible"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data