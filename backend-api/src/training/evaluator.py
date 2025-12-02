"""
Evaluator pour évaluer les modèles Recycle-moi
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from ..config import config

class Evaluator:
    """Classe pour évaluer les modèles"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: List[str] = None,
        device: str = None
    ):
        """
        Initialise l'evaluator
        
        Args:
            model: Modèle à évaluer
            test_loader: DataLoader de test
            class_names: Noms des classes
            device: Device (cuda ou cpu)
        """
        self.model = model
        self.test_loader = test_loader
        
        # Class names
        if class_names is None:
            class_names = config.get('data.classes')
        self.class_names = class_names
        
        # Device
        if device is None:
            device = config.get('hardware.device', 'cuda')
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, criterion: nn.Module = None) -> Dict:
        """
        Évalue le modèle sur le test set
        
        Args:
            criterion: Fonction de loss (optionnel)
            
        Returns:
            Dictionnaire de métriques
        """
        all_preds = []
        all_labels = []
        all_probs = []
        running_loss = 0.0
        
        print("🧪 ÉVALUATION SUR LE TEST SET")
        print("=" * 60)
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Évaluation')
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                outputs = self.model(images)
                
                # Loss
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)
                
                # Prédictions
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Sauvegarder
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculer métriques
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        
        results = {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        if criterion is not None:
            results['loss'] = running_loss / len(self.test_loader.dataset)
        
        # Affichage
        print(f"\n🏆 RÉSULTATS FINAUX")
        print("=" * 60)
        print(f"Test Accuracy : {accuracy:.2f}%")
        if criterion is not None:
            print(f"Test Loss : {results['loss']:.4f}")
        print("=" * 60)
        
        return results
    
    def get_confusion_matrix(self, results: Dict = None) -> np.ndarray:
        """
        Calcule la matrice de confusion
        
        Args:
            results: Résultats de evaluate() (si None, évalue d'abord)
            
        Returns:
            Matrice de confusion
        """
        if results is None:
            results = self.evaluate()
        
        cm = confusion_matrix(results['labels'], results['predictions'])
        return cm
    
    def get_classification_report(self, results: Dict = None) -> str:
        """
        Génère un rapport de classification détaillé
        
        Args:
            results: Résultats de evaluate() (si None, évalue d'abord)
            
        Returns:
            Rapport de classification
        """
        if results is None:
            results = self.evaluate()
        
        report = classification_report(
            results['labels'],
            results['predictions'],
            target_names=self.class_names,
            digits=4
        )
        
        return report
    
    def evaluate_per_class(self, results: Dict = None) -> Dict:
        """
        Calcule les métriques par classe
        
        Args:
            results: Résultats de evaluate() (si None, évalue d'abord)
            
        Returns:
            Dictionnaire de métriques par classe
        """
        if results is None:
            results = self.evaluate()
        
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            # Indices de cette classe
            class_indices = results['labels'] == i
            
            if class_indices.sum() > 0:
                # Accuracy pour cette classe
                class_accuracy = (results['predictions'][class_indices] == i).mean() * 100
                
                # Nombre d'échantillons
                num_samples = class_indices.sum()
                
                per_class_metrics[class_name] = {
                    'accuracy': class_accuracy,
                    'num_samples': int(num_samples)
                }
        
        return per_class_metrics
    
    def print_detailed_results(self, results: Dict = None) -> None:
        """
        Affiche des résultats détaillés
        
        Args:
            results: Résultats de evaluate() (si None, évalue d'abord)
        """
        if results is None:
            results = self.evaluate()
        
        print("\n📊 RAPPORT DE CLASSIFICATION DÉTAILLÉ")
        print("=" * 60)
        print(self.get_classification_report(results))
        
        print("\n📈 MÉTRIQUES PAR CLASSE")
        print("=" * 60)
        per_class = self.evaluate_per_class(results)
        
        for class_name, metrics in per_class.items():
            print(f"{class_name:12s}: {metrics['accuracy']:6.2f}% ({metrics['num_samples']:4d} échantillons)")
        
        print("=" * 60)