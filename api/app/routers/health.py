from fastapi import APIRouter, Depends
from ..services.health_service import HealthService
from ..schemas.health import HealthResponse

router = APIRouter(tags=["health"])

def get_health_service():
    return HealthService()

@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Image Similarity Search API",
        "docs": "/docs"
    }

@router.get("/health", response_model=HealthResponse)
async def health_check(service: HealthService = Depends(get_health_service)):
    """Detailed health check"""
    return service.get_health_status()
