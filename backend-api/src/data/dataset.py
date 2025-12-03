"""
Dataset et DataLoaders pour Recycle-moi
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
from typing import Tuple

from .transforms import get_train_transforms, get_eval_transforms
from ..config import config

def create_dataloaders(
    data_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
    pin_memory: bool = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les DataLoaders pour train, valid et test
    
    Args:
        data_dir: Chemin vers le dataset (défaut: depuis config)
        batch_size: Taille du batch (défaut: depuis config)
        num_workers: Nombre de workers (défaut: depuis config)
        pin_memory: Pin memory pour GPU (défaut: depuis config)
        
    Returns:
        Tuple de (train_loader, valid_loader, test_loader)
    """
    # Récupérer les paramètres depuis config si non fournis
    if data_dir is None:
        data_dir = config.get('data.data_dir')
    if batch_size is None:
        batch_size = config.get('training.batch_size', 32)
    if num_workers is None:
        num_workers = config.get('hardware.num_workers', 4)
    if pin_memory is None:
        pin_memory = config.get('hardware.pin_memory', True)
    
    data_dir = Path(data_dir)
    
    # Récupérer les stats de normalisation
    mean = config.get('data.mean')
    std = config.get('data.std')
    
    # Transformations
    train_transforms = get_train_transforms(mean=mean, std=std)
    eval_transforms = get_eval_transforms(mean=mean, std=std)
    
    # Datasets
    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=train_transforms
    )
    
    valid_dataset = datasets.ImageFolder(
        data_dir / "valid",
        transform=eval_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        data_dir / "test",
        transform=eval_transforms
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, valid_loader, test_loader

def get_class_names(data_dir: str = None) -> list:
    """
    Récupère les noms des classes depuis le dataset
    
    Args:
        data_dir: Chemin vers le dataset
        
    Returns:
        Liste des noms de classes
    """
    if data_dir is None:
        data_dir = config.get('data.data_dir')
    
    data_dir = Path(data_dir)
    train_dataset = datasets.ImageFolder(data_dir / "train")
    
    return train_dataset.classes