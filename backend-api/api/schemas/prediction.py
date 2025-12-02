"""
Schémas Pydantic pour la validation des requêtes/réponses
"""

from pydantic import BaseModel, Field
from typing import List, Dict

class PredictionResponse(BaseModel):
    """
    Réponse de l'API pour une prédiction
    """
    success: bool = Field(..., description="Statut de la prédiction")
    predicted_class: str = Field(..., description="Classe prédite")
    confidence: float = Field(..., description="Confiance de la prédiction (0-1)", ge=0, le=1)
    all_probabilities: Dict[str, float] = Field(..., description="Probabilités pour toutes les classes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predicted_class": "plastic",
                "confidence": 0.89,
                "all_probabilities": {
                    "cardboard": 0.02,
                    "e-waste": 0.01,
                    "glass": 0.03,
                    "medical": 0.01,
                    "metal": 0.02,
                    "paper": 0.02,
                    "plastic": 0.89
                }
            }
        }

class ErrorResponse(BaseModel):
    """
    Réponse en cas d'erreur
    """
    success: bool = False
    error: str = Field(..., description="Message d'erreur")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid image format"
            }
        }

class ModelInfoResponse(BaseModel):
    """
    Informations sur le modèle
    """
    model_version: str
    architecture: str
    num_classes: int
    classes: List[str]
    test_accuracy: float
    created_at: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_version": "1.0",
                "architecture": "resnet18",
                "num_classes": 7,
                "classes": ["cardboard", "e-waste", "glass", "medical", "metal", "paper", "plastic"],
                "test_accuracy": 83.55,
                "created_at": "2024-12-02T14:30:00"
            }
        }