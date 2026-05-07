from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Database (from .env)
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    
    # API (from .env)
    api_host: str
    api_port: int
    api_title: str = "Image Similarity Search API"
    api_description: str = "Upload images and search for similar ones (Modular Architecture)"
    api_version: str = "2.2.0"
    
    # AI & Models
    hf_token: str = "" # Use HF_TOKEN in .env
    openrouter_api_key: str = ""
    llm_vision_model: str = "google/gemini-2.0-flash-001"
    llm_vision_local_model: str = "microsoft/Florence-2-large"
    llm_embedding_model: str = "BAAI/bge-m3"
    clip_model_name: str = "openai/clip-vit-large-patch14"
    dinov2_model_name: str = "facebook/dinov2-giant"
    siglip_model_name: str = "google/siglip-base-patch16-224"
    convnext_model_name: str = "facebook/convnextv2-base-1k-224"
    efficientnet_model_name: str = "google/efficientnet-b7"
    sam_model_name: str = "facebook/sam-vit-base"
    
    # CORS
    cors_origins: List[str] = ["http://localhost:5173"]
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    uploads_dir: Path = base_dir / "uploads"
    visualizations_dir: Path = base_dir / "visualizations"
    weights_file: Path = base_dir / "weights.json"
    evaluation_results_file: Path = base_dir / "evaluation_results.json"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = "app/.env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env
    
    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    def setup_directories(self):
        """Create necessary directories"""
        self.uploads_dir.mkdir(exist_ok=True, parents=True)
        self.visualizations_dir.mkdir(exist_ok=True, parents=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)"""
    settings = Settings()
    # Tạo thư mục uploads
    settings.setup_directories()
    return settings
