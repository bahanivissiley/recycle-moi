"""
Transformations d'images pour Recycle-moi
"""

from torchvision import transforms
from typing import Tuple, List

def get_train_transforms(
    img_size: int = 224,
    mean: List[float] = [0.649, 0.630, 0.602],
    std: List[float] = [0.211, 0.212, 0.222]
) -> transforms.Compose:
    """
    Transformations pour l'entraînement (avec augmentation)
    
    Args:
        img_size: Taille de l'image (défaut 224)
        mean: Moyenne pour normalisation
        std: Écart-type pour normalisation
        
    Returns:
        Composition de transformations
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_eval_transforms(
    img_size: int = 224,
    mean: List[float] = [0.649, 0.630, 0.602],
    std: List[float] = [0.211, 0.212, 0.222]
) -> transforms.Compose:
    """
    Transformations pour validation/test (sans augmentation)
    
    Args:
        img_size: Taille de l'image (défaut 224)
        mean: Moyenne pour normalisation
        std: Écart-type pour normalisation
        
    Returns:
        Composition de transformations
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_inference_transforms(
    img_size: int = 224,
    mean: List[float] = [0.649, 0.630, 0.602],
    std: List[float] = [0.211, 0.212, 0.222]
) -> transforms.Compose:
    """
    Transformations pour inférence (API)
    Identique à eval mais explicite pour clarté
    
    Args:
        img_size: Taille de l'image (défaut 224)
        mean: Moyenne pour normalisation
        std: Écart-type pour normalisation
        
    Returns:
        Composition de transformations
    """
    return get_eval_transforms(img_size, mean, std)