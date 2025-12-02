"""
Fonctions utilitaires pour Recycle-moi
"""

import torch
import random
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

def set_seed(seed: int = 42) -> None:
    """
    Fixe la seed pour la reproductibilité
    
    Args:
        seed: Valeur de la seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Seed fixée à {seed}")

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Compte les paramètres d'un modèle
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Dictionnaire avec total et trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }

def save_metadata(
    save_path: str,
    model_info: Dict[str, Any],
    training_info: Dict[str, Any],
    results: Dict[str, Any]
) -> None:
    """
    Sauvegarde les métadonnées du modèle
    
    Args:
        save_path: Chemin de sauvegarde (JSON)
        model_info: Infos sur le modèle
        training_info: Infos sur l'entraînement
        results: Résultats finaux
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model': model_info,
        'training': training_info,
        'results': results
    }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Métadonnées sauvegardées : {save_path}")

def load_metadata(metadata_path: str) -> Dict:
    """
    Charge les métadonnées d'un modèle
    
    Args:
        metadata_path: Chemin vers le fichier JSON
        
    Returns:
        Dictionnaire de métadonnées
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def format_time(seconds: float) -> str:
    """
    Formate un temps en secondes en format lisible
    
    Args:
        seconds: Temps en secondes
        
    Returns:
        String formaté (ex: "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"