from typing import Generator
from sqlalchemy.orm import Session
from functools import lru_cache

from ..services import DatabaseService, ImageProcessor


# Service instances (singleton pattern)
_db_service: DatabaseService = None
_image_processor: ImageProcessor = None


def get_database_service() -> DatabaseService:
    """Get database service instance"""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service


def get_image_processor() -> ImageProcessor:
    """Get image processor instance"""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor


def get_db() -> Generator[Session, None, None]:
    db_service = get_database_service()
    db = db_service.get_session()
    try:
        yield db
    finally:
        db.close()
