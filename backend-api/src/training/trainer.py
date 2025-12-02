"""
Trainer pour l'entraînement des modèles Recycle-moi
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

from ..config import config

class Trainer:
    """Classe pour entraîner les modèles"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = None,
        checkpoint_dir: str = None
    ):
        """
        Initialise le trainer
        
        Args:
            model: Modèle à entraîner
            train_loader: DataLoader d'entraînement
            valid_loader: DataLoader de validation
            criterion: Fonction de loss
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optionnel)
            device: Device (cuda ou cpu)
            checkpoint_dir: Dossier pour sauvegarder les checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Device
        if device is None:
            device = config.get('hardware.device', 'cuda')
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = config.get('paths.checkpoints', 'checkpoints')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Historique
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_acc': [],
            'epoch_times': []
        }
        
        # Meilleure accuracy
        self.best_valid_acc = 0.0
        self.best_model_path = None
    
    def train_one_epoch(self) -> Tuple[float, float]:
        """
        Entraîne le modèle sur une epoch
        
        Returns:
            Tuple de (loss moyenne, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for images, labels in pbar:
            # Vers GPU
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistiques
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Mise à jour barre de progression
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """
        Évalue le modèle sur le validation set
        
        Returns:
            Tuple de (loss moyenne, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc='Validation', leave=False)
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistiques
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self,
        num_epochs: int = None,
        save_best: bool = True,
        early_stopping_patience: int = None
    ) -> Dict:
        """
        Entraîne le modèle sur plusieurs epochs
        
        Args:
            num_epochs: Nombre d'epochs (défaut: depuis config)
            save_best: Sauvegarder le meilleur modèle
            early_stopping_patience: Patience pour early stopping (défaut: depuis config)
            
        Returns:
            Dictionnaire de l'historique d'entraînement
        """
        if num_epochs is None:
            num_epochs = config.get('training.num_epochs', 10)
        
        if early_stopping_patience is None:
            early_stopping_patience = config.get('training.patience', 5)
        
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 60)
        
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # TRAIN
            train_loss, train_acc = self.train_one_epoch()
            
            # VALID
            valid_loss, valid_acc = self.validate()
            
            # Temps epoch
            epoch_time = time.time() - epoch_start
            
            # Sauvegarder historique
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['valid_loss'].append(valid_loss)
            self.history['valid_acc'].append(valid_acc)
            self.history['epoch_times'].append(epoch_time)
            
            # Affichage
            print(f"\n📊 Résultats Epoch {epoch+1}:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%")
            print(f"   ⏱️  Temps : {epoch_time/60:.1f} min")
            
            # Sauvegarder meilleur modèle
            if save_best and valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                self.best_model_path = self.checkpoint_dir / 'best_model.pth'
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"   ✅ Meilleur modèle sauvegardé! (Acc: {self.best_valid_acc:.2f}%)")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️  Early stopping après {epoch+1} epochs (pas d'amélioration depuis {early_stopping_patience} epochs)")
                break
            
            print("-" * 60)
        
        # Stats finales
        total_time = time.time() - start_time
        print(f"\n⏱️  Temps total : {total_time/60:.1f} minutes")
        print(f"🏆 Meilleure validation accuracy : {self.best_valid_acc:.2f}%")
        print("=" * 60)
        
        return self.history
    
    def get_history(self) -> Dict:
        """Retourne l'historique d'entraînement"""
        return self.history
    
    def save_checkpoint(self, epoch: int, filename: str = None) -> None:
        """
        Sauvegarde un checkpoint complet
        
        Args:
            epoch: Numéro de l'epoch
            filename: Nom du fichier (défaut: checkpoint_epoch_{epoch}.pth)
        """
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_valid_acc': self.best_valid_acc
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint sauvegardé : {checkpoint_path}")