"""
Route de health check
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Vérifie que l'API est en ligne
    
    Returns:
        Status de l'API
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Recycle-moi API"
    }