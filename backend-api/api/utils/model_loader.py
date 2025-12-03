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
    
    def load(self, checkpoint_path: str, metadata_path: str = None):
        """
        Charge le modèle et ses métadonnées
        
        Args:
            checkpoint_path: Chemin vers le fichier .pth
            metadata_path: Chemin vers metadata.json (optionnel)
        """
        if self._model is not None:
            print("⚠️  Modèle déjà chargé")
            return
        
        print("🔄 Chargement du modèle...")
        
        # Device
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {self._device}")
        
        # Charger le modèle
        self._model = load_model(checkpoint_path, device=str(self._device))
        self._model.eval()
        print(f"   ✅ Modèle chargé: {checkpoint_path}")
        
        # Charger métadonnées
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self._metadata = json.load(f)
            print(f"   ✅ Métadonnées chargées: {metadata_path}")
        else:
            # Métadonnées par défaut depuis config
            self._metadata = {
                'version': '1.0',
                'architecture': 'resnet18',
                'num_classes': config.get('data.num_classes'),
                'classes': config.get('data.classes'),
                'results': {
                    'test_accuracy': 83.55
                }
            }
            print("   ⚠️  Métadonnées par défaut utilisées")
        
        # Classes
        self._classes = self._metadata.get('data', {}).get('classes') or config.get('data.classes')
        
        # Transforms
        mean = config.get('data.mean')
        std = config.get('data.std')
        self._transforms = transforms.get_inference_transforms(mean=mean, std=std)
        print(f"   ✅ Transforms configurés")
        
        print("✅ Modèle prêt pour les prédictions")
    
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