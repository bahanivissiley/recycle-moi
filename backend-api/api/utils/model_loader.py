"""
Chargeur de modèle singleton pour l'API
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
import os
import requests
from typing import Optional, Dict, Any, List
from PIL import Image

from src.data import transforms
from src.models.resnet import create_resnet18  # ← create au lieu de load
from src.config import config


def download_model_from_url(url: str, checkpoint_path: str):
    """Télécharge le modèle depuis une URL si absent"""
    if Path(checkpoint_path).exists():
        print(f"✅ Modèle déjà présent : {checkpoint_path}")
        return
    
    print(f"📥 Téléchargement du modèle depuis {url}...")
    
    try:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(checkpoint_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        print(f"   📦 {mb_downloaded:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)", end='\r')
        
        print(f"\n✅ Modèle téléchargé : {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Erreur téléchargement : {e}")
        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
        raise


class ModelLoader:
    """Singleton pour charger et gérer le modèle"""
    
    _instance: Optional['ModelLoader'] = None
    _model: Optional[torch.nn.Module] = None
    _metadata: Optional[Dict[str, Any]] = None
    _transforms = None
    _device = None
    _classes: List[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, checkpoint_path: str, metadata_path: str = None):
        """Charge le modèle et ses métadonnées"""
        
        if self._model is not None:
            print("⚠️  Modèle déjà chargé")
            return
        
        print("🔄 Chargement du modèle...")
        
        # URL du modèle
        model_url = os.getenv(
            'MODEL_URL',
            'https://huggingface.co/bahani/recyclemoi-resnet18/resolve/main/best_model.pth'
        )
        
        print(f"📍 URL : {model_url}")
        
        # Télécharger le modèle
        download_model_from_url(model_url, checkpoint_path)
        
        print("🔄 Chargement dans PyTorch...")
        
        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {self._device}")
        
        # Classes
        self._classes = getattr(config, 'CLASSES', [
            'cardboard', 'e-waste', 'glass', 'medical', 'metal', 'paper', 'plastic'
        ])
        num_classes = len(self._classes)
        
        print(f"   Classes ({num_classes}): {', '.join(self._classes)}")
        
        # Créer et charger le modèle
        try:
            # Créer l'architecture
            self._model = create_resnet18(
                num_classes=num_classes,
                pretrained=False
            )
            print("   ✅ Architecture créée")
            
            # Charger les poids
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            self._model.load_state_dict(checkpoint)
            self._model.to(self._device)
            self._model.eval()
            print("   ✅ Poids chargés")
            
        except Exception as e:
            print(f"   ❌ Erreur chargement: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Transforms
        try:
            self._transforms = transforms.get_transforms()["test"]
            print("   ✅ Transforms configurés")
        except Exception as e:
            print(f"   ⚠️  Transforms par défaut: {e}")
            from torchvision import transforms as T
            self._transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.649, 0.630, 0.602], std=[0.211, 0.212, 0.222])
            ])
        
        # Métadonnées
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)
            print("   ✅ Métadonnées chargées")
        else:
            self._metadata = {
                "version": "1.0",
                "model": {
                    "architecture": "resnet18",
                    "num_classes": num_classes
                },
                "data": {
                    "classes": self._classes,
                    "num_classes": num_classes
                },
                "results": {
                    "test_accuracy": 83.55
                },
                "created_at": "2024-12-02"
            }
            print("   ℹ️  Métadonnées par défaut")
        
        print("✅ Modèle prêt !")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Prédiction sur une image"""
        
        if self._model is None:
            raise RuntimeError("Modèle non chargé")
        
        # Prétraitement
        image_tensor = self._transforms(image).unsqueeze(0).to(self._device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self._model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        # Résultat
        predicted_class = self._classes[predicted_idx]
        all_probs = {
            self._classes[i]: float(probabilities[i].item())
            for i in range(len(self._classes))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Retourne les métadonnées"""
        if self._metadata is None:
            raise RuntimeError("Métadonnées non chargées")
        return self._metadata
    
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé"""
        return self._model is not None


# Instance globale
model_loader = ModelLoader()