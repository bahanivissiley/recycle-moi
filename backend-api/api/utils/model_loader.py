import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
from typing import Optional, Dict, Any
from PIL import Image

from src.data import transforms
from src.models.resnet import load_model
from src.config import config


import os
import requests
from pathlib import Path

def download_model_from_url(url: str, checkpoint_path: str):
    """
    Télécharge le modèle depuis une URL si absent
    
    Args:
        url: URL du modèle
        checkpoint_path: Chemin local où sauvegarder
    """
    if Path(checkpoint_path).exists():
        print(f"✅ Modèle déjà présent : {checkpoint_path}")
        return
    
    print(f"📥 Téléchargement du modèle depuis {url}...")
    
    try:
        # Créer le dossier si nécessaire
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Télécharger avec progress
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(checkpoint_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        print(f"   📦 {mb_downloaded:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)", end='\r')
        
        print(f"\n✅ Modèle téléchargé : {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        # Supprimer le fichier partiel
        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
        raise
    


class ModelLoader:
    """
    Singleton pour charger et gérer le modèle
    """
    _instance: Optional['ModelLoader'] = None
    _model: Optional[torch.nn.Module] = None
    _metadata: Optional[Dict[str, Any]] = None
    _transforms = None
    _device = None
    _classes = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    

    # Dans la classe ModelLoader, méthode load()
    def load(self, checkpoint_path: str, metadata_path: str = None):
        """Charge le modèle"""
        if self._model is not None:
            print("⚠️  Modèle déjà chargé")
            return
        
        print("🔄 Chargement du modèle...")
        
        # URL du modèle (depuis variable d'environnement ou défaut)
        model_url = os.getenv(
            'MODEL_URL',
            'https://huggingface.co/bahani/recyclemoi-resnet18/blob/main/best_model.pth'
        )
        
        # Télécharger si absent
        download_model_from_url(model_url, checkpoint_path)

    
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Fait une prédiction sur une image
        
        Args:
            image: Image PIL
            
        Returns:
            Dictionnaire avec la prédiction
        """
        if self._model is None:
            raise RuntimeError("Modèle non chargé. Appelez load() d'abord.")
        
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
        """Retourne les métadonnées du modèle"""
        if self._metadata is None:
            raise RuntimeError("Métadonnées non chargées")
        return self._metadata
    
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé"""
        return self._model is not None

# Instance globale
model_loader = ModelLoader()