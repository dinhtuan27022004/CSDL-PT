"""
Main Application Entry Point
FastAPI application configuration and startup
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import get_settings
from api.routes import health_router, images_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    
    Returns:
        Configured FastAPI app instance
    """
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    app.mount(
        "/static/uploads",
        StaticFiles(directory=str(settings.uploads_dir)),
        name="uploads"
    )
    
    # Register routes
    app.include_router(health_router)
    app.include_router(images_router)
    
    logger.info("FastAPI application created successfully")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    print("\n" + "=" * 60)
    print("Image Similarity Search API Server (OOP Architecture)")
    print("=" * 60)
    print(f"\nDatabase: {settings.database_url}")
    print(f"Uploads: {settings.uploads_dir}")
    print(f"\nAPI URL: http://localhost:{settings.api_port}")
    print(f"API Docs: http://localhost:{settings.api_port}/docs")
    print(f"Redoc: http://localhost:{settings.api_port}/redoc")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )

# Trigger reload 2
