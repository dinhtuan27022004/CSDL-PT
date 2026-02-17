"""Routes module"""

from .health import router as health_router
from .images import router as images_router

__all__ = ["health_router", "images_router"]
