"""
Script d'évaluation pour Recycle-moi
Usage: python scripts/evaluate.py --checkpoint checkpoints/v1.0/best_model.pth
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.config import config
from src.data.dataset import create_dataloaders, get_class_names
from src.models.resnet import load_model
from src.training.evaluator import Evaluator
from src.utils.logger import setup_logger
from src.utils.helpers import load_metadata

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Évaluation du modèle Recycle-moi')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle (.pth)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Taille du batch (défaut: depuis config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda ou cpu, défaut: depuis config)')
    parser.add_argument('--save-confusion-matrix', action='store_true',
                       help='Sauvegarder la matrice de confusion')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Dossier pour sauvegarder les résultats')
    
    return parser.parse_args()

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Affiche et sauvegarde la matrice de confusion
    
    Args:
        cm: Matrice de confusion
        class_names: Noms des classes
        save_path: Chemin de sauvegarde (optionnel)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Nombre de prédictions'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.xlabel('Prédictions', fontsize=12, fontweight='bold')
    plt.ylabel('Vraies classes', fontsize=12, fontweight='bold')
    plt.title('Matrice de Confusion - Test Set', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matrice de confusion sauvegardée : {save_path}")
    
    plt.show()

def main():
    """Fonction principale d'évaluation"""
    
    # Parser arguments
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(name='recyclemoi_eval')
    logger.info("🧪 Démarrage de l'évaluation Recycle-moi")
    
    # Device
    device = args.device if args.device else config.get('hardware.device', 'cuda')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")
    
    # Vérifier que le checkpoint existe
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint non trouvé : {checkpoint_path}")
        return
    
    logger.info(f"📦 Checkpoint : {checkpoint_path}")
    
    # Charger métadonnées si disponibles
    metadata_path = checkpoint_path.parent / 'metadata.json'
    if metadata_path.exists():
        logger.info(f"📋 Chargement des métadonnées...")
        metadata = load_metadata(metadata_path)
        logger.info(f"   Modèle entraîné le : {metadata.get('timestamp', 'N/A')}")
        logger.info(f"   Architecture : {metadata.get('model', {}).get('architecture', 'N/A')}")
        logger.info(f"   Best valid acc : {metadata.get('results', {}).get('best_valid_acc', 'N/A'):.2f}%")
    
    # ==================== DATA ====================
    logger.info("📊 Chargement des données de test...")
    
    _, _, test_loader = create_dataloaders(batch_size=args.batch_size)
    class_names = get_class_names()
    
    logger.info(f"   Test: {len(test_loader)} batches ({len(test_loader.dataset)} images)")
    logger.info(f"   Classes: {class_names}")
    
    # ==================== MODEL ====================
    logger.info("🏗️  Chargement du modèle...")
    
    model = load_model(
        checkpoint_path=str(checkpoint_path),
        device=str(device)
    )
    
    logger.info("   Modèle chargé avec succès")
    
    # ==================== EVALUATION ====================
    logger.info("🧪 Évaluation du modèle...")
    
    criterion = nn.CrossEntropyLoss()
    
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=str(device)
    )
    
    # Évaluer
    results = evaluator.evaluate(criterion=criterion)
    
    # Afficher rapport détaillé
    evaluator.print_detailed_results(results)
    
    # ==================== CONFUSION MATRIX ====================
    if args.save_confusion_matrix:
        logger.info("📊 Génération de la matrice de confusion...")
        
        cm = evaluator.get_confusion_matrix(results)
        
        # Créer dossier output
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder
        cm_path = output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(cm, class_names, save_path=cm_path)
    
    # ==================== SAVE RESULTS ====================
    logger.info("💾 Sauvegarde des résultats...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder rapport
    report = evaluator.get_classification_report(results)
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"   Rapport : {report_path}")
    
    # Sauvegarder métriques par classe
    import json
    per_class = evaluator.evaluate_per_class(results)
    per_class_path = output_dir / 'per_class_metrics.json'
    with open(per_class_path, 'w') as f:
        json.dump(per_class, f, indent=2)
    logger.info(f"   Métriques par classe : {per_class_path}")
    
    logger.info("✅ Évaluation terminée avec succès!")
    logger.info(f"🏆 Test Accuracy finale : {results['accuracy']:.2f}%")

if __name__ == '__main__':
    main()