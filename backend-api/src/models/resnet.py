"""
Modèle ResNet pour Recycle-moi
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

from ..config import config

def create_resnet18(
    num_classes: int = None,
    pretrained: bool = None,
    freeze_backbone: bool = None
) -> nn.Module:
    """
    Crée un modèle ResNet18 pour la classification de déchets
    
    Args:
        num_classes: Nombre de classes (défaut: depuis config)
        pretrained: Utiliser les poids ImageNet (défaut: depuis config)
        freeze_backbone: Geler les couches convolutionnelles (défaut: depuis config)
        
    Returns:
        Modèle ResNet18 configuré
    """
    # Récupérer depuis config si non fourni
    if num_classes is None:
        num_classes = config.get('data.num_classes', 7)
    if pretrained is None:
        pretrained = config.get('model.pretrained', True)
    if freeze_backbone is None:
        freeze_backbone = config.get('model.freeze_backbone', False)
    
    # Charger ResNet18
    if pretrained:
        model = models.resnet18(weights='IMAGENET1K_V1')
    else:
        model = models.resnet18(weights=None)
    
    # Geler les couches si demandé
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Remplacer la dernière couche pour nos classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def load_model(
    checkpoint_path: str,
    num_classes: int = None,
    device: str = 'cuda'
) -> nn.Module:
    """
    Charge un modèle depuis un checkpoint
    
    Args:
        checkpoint_path: Chemin vers le fichier .pth
        num_classes: Nombre de classes
        device: Device (cuda ou cpu)
        
    Returns:
        Modèle chargé
    """
    if num_classes is None:
        num_classes = config.get('data.num_classes', 7)
    
    # Créer le modèle
    model = create_resnet18(num_classes=num_classes, pretrained=False)
    
    # Charger les poids
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    model = model.to(device)
    model.eval()
    
    return model

def save_model(
    model: nn.Module,
    save_path: str
) -> None:
    """
    Sauvegarde un modèle
    
    Args:
        model: Modèle à sauvegarder
        save_path: Chemin de sauvegarde
    """
    torch.save(model.state_dict(), save_path)
    print(f"✅ Modèle sauvegardé : {save_path}")