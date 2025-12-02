"""
Module de configuration pour Recycle-moi
Charge et expose les paramètres depuis config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Classe pour charger et accéder à la configuration"""
    
    def __init__(self, config_path: str = None):
        """
        Charge la configuration depuis un fichier YAML
        
        Args:
            config_path: Chemin vers le fichier config.yaml
        """
        if config_path is None:
            # Par défaut, cherche config.yaml dans le même dossier
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de config avec notation pointée
        
        Args:
            key: Clé au format 'section.subsection.key'
            default: Valeur par défaut si clé non trouvée
            
        Returns:
            La valeur de config ou default
            
        Example:
            >>> config.get('data.batch_size')
            32
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Permet d'accéder à la config avec config['key']"""
        return self.get(key)
    
    @property
    def data(self) -> Dict:
        """Retourne la section data"""
        return self._config.get('data', {})
    
    @property
    def model(self) -> Dict:
        """Retourne la section model"""
        return self._config.get('model', {})
    
    @property
    def training(self) -> Dict:
        """Retourne la section training"""
        return self._config.get('training', {})
    
    @property
    def hardware(self) -> Dict:
        """Retourne la section hardware"""
        return self._config.get('hardware', {})
    
    @property
    def paths(self) -> Dict:
        """Retourne la section paths"""
        return self._config.get('paths', {})

# Instance globale de config
config = Config()