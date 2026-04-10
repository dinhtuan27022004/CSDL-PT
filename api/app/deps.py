from typing import Generator
from .db.session import SessionLocal

def get_db() -> Generator:
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
