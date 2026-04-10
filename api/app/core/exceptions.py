from fastapi import HTTPException, status
from typing import Any, Dict, Optional

class AppException(Exception):
    """Base class for application exceptions"""
    def __init__(
        self, 
        message: str, 
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: Optional[Any] = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)

class ImageProcessingError(AppException):
    """Raised when image processing fails"""
    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(
            message=message, 
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

class DatabaseError(AppException):
    """Raised when database operations fail"""
    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(
            message=message, 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

class EntityNotFoundError(AppException):
    """Raised when an entity is not found"""
    def __init__(self, entity_name: str, entity_id: Any):
        super().__init__(
            message=f"{entity_name} with id {entity_id} not found", 
            status_code=status.HTTP_404_NOT_FOUND
        )
