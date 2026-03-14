"""
Configuration Management
Centralized settings using Pydantic BaseSettings
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5434
    db_name: str = "csdldpt"
    db_user: str = "postgres"
    db_password: str = "123123"
    
    # API
    api_title: str = "Image Similarity Search API"
    api_description: str = "Upload images and search for similar ones"
    api_version: str = "2.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # CORS
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    uploads_dir: Path = base_dir / "uploads"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env
    
    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    def setup_directories(self):
        """Create necessary directories"""
        self.uploads_dir.mkdir(exist_ok=True, parents=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)"""
    settings = Settings()
    settings.setup_directories()
    return settings
