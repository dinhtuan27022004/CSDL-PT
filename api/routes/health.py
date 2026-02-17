"""
Health Check Routes
Simple status endpoints
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Image Similarity Search API",
        "docs": "/docs"
    }


@router.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "Image Similarity Search API",
        "version": "2.0.0"
    }
