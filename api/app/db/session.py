from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

engine = create_engine(
    settings.database_url, 
    echo=False,
    pool_size=20,          # Increase base pool from 5 to 20
    max_overflow=30,       # Allow up to 30 extra connections
    pool_timeout=60,       # Wait up to 60s for a connection
    pool_recycle=1800      # Recycle connections every 30 mins
)

SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

def init_db():
    """Ensure vector extension and tables exist"""
    try:
        from .base import Base
        # Import models here to ensure they are registered with Base.metadata before create_all
        from ..models.image import ImageMetadata
        
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
