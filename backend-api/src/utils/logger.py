"""
Logger pour Recycle-moi
"""

import logging
from pathlib import Path
from datetime import datetime
from ..config import config

def setup_logger(
    name: str = 'recyclemoi',
    log_dir: str = None,
    level: str = None
) -> logging.Logger:
    """
    Configure un logger
    
    Args:
        name: Nom du logger
        log_dir: Dossier pour les logs
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Logger configuré
    """
    # Récupérer depuis config si non fourni
    if log_dir is None:
        log_dir = config.get('paths.logs', 'logs')
    if level is None:
        level = config.get('logging.level', 'INFO')
    
    # Créer dossier logs
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Nom du fichier log avec timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'
    
    # Créer logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Éviter les doublons de handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler fichier
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialisé : {log_file}")
    
    return logger