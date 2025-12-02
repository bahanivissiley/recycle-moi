"""
Routes pour les informations du modèle
"""

from fastapi import APIRouter, HTTPException
from api.schemas.prediction import ModelInfoResponse, ErrorResponse
from api.utils.model_loader import model_loader

router = APIRouter()

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Retourne les informations sur le modèle chargé
    
    Returns:
        Informations du modèle (version, architecture, accuracy, etc.)
    """
    try:
        if not model_loader.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
        
        metadata = model_loader.get_metadata()
        
        # Extraire les infos importantes
        response = {
            "model_version": metadata.get('version', 'unknown'),
            "architecture": metadata.get('model', {}).get('architecture', 'resnet18'),
            "num_classes": metadata.get('data', {}).get('num_classes', 7),
            "classes": metadata.get('data', {}).get('classes', []),
            "test_accuracy": metadata.get('results', {}).get('test_accuracy', 0.0),
            "created_at": metadata.get('created_at', 'unknown')
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model info: {str(e)}"
        )