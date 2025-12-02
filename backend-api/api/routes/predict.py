"""
Routes pour les prédictions
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io
from typing import Union

from api.schemas.prediction import PredictionResponse, ErrorResponse
from api.utils.model_loader import model_loader

router = APIRouter()

# Formats d'image acceptés
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_image(file: UploadFile) -> bool:
    """
    Valide le fichier uploadé
    
    Args:
        file: Fichier uploadé
        
    Returns:
        True si valide, sinon lève une exception
    """
    # Vérifier l'extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    return True

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Prédit la classe d'un déchet à partir d'une image
    
    Args:
        file: Image uploadée (PNG, JPG, JPEG, WEBP, BMP)
        
    Returns:
        Prédiction avec classe, confiance et probabilités pour toutes les classes
        
    Raises:
        HTTPException: Si l'image est invalide ou si une erreur survient
    """
    try:
        # Vérifier que le modèle est chargé
        if not model_loader.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please try again later."
            )
        
        # Valider le fichier
        validate_image(file)
        
        # Lire le fichier
        contents = await file.read()
        
        # Vérifier la taille
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Charger l'image
        try:
            image = Image.open(io.BytesIO(contents))
            
            # Convertir en RGB si nécessaire (pour les PNG avec transparence, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Faire la prédiction
        try:
            prediction = model_loader.predict(image)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
        
        # Retourner la réponse
        return {
            "success": True,
            "predicted_class": prediction['predicted_class'],
            "confidence": prediction['confidence'],
            "all_probabilities": prediction['all_probabilities']
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/predict/url")
async def predict_from_url(image_url: str):
    """
    Prédit à partir d'une URL d'image
    (Fonctionnalité future - non implémentée pour l'instant)
    """
    raise HTTPException(
        status_code=501,
        detail="URL prediction not implemented yet. Please use file upload."
    )