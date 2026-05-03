from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path

# Add current directory to path for relative imports if needed
sys.path.append(str(Path(__file__).parent))

from .core.config import get_settings
from .core.logging import setup_logging, get_logger
from .db.session import init_db
from .routers.health import router as health_router
from .routers.image import router as images_router

logger = get_logger(__name__)

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    setup_logging()
    
    init_db()
    
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Relaxed for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    settings.setup_directories()
    app.mount(
        "/static/uploads",
        StaticFiles(directory=str(settings.uploads_dir)),
        name="uploads"
    )
    app.mount(
        "/static/visualizations",
        StaticFiles(directory=str(settings.visualizations_dir)),
        name="visualizations"
    )
    
    # Register routes
    app.include_router(health_router)
    app.include_router(images_router)
    
    logger.info("FastAPI modular application created successfully")
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
