import os
import asyncio
import time
from pathlib import Path
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.services.image_service import ImageService
from app.repositories.image_repository import ImageRepository
from app.services.llm_service import LLMService
from app.services.cache_service import CacheService
from app.core.logging import setup_logging, get_logger
from app.core.config import get_settings

setup_logging()
logger = get_logger("import_script")
settings = get_settings()

async def import_from_folder(folder_path: str, batch_size: int = 50):
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder {folder_path} does not exist!")
        return

    # Initialize services
    repo = ImageRepository()
    cache = CacheService()
    llm = LLMService(cache_service=cache)
    service = ImageService(repository=repo, llm_service=llm, cache_service=cache)
    
    db = SessionLocal()
    try:
        # 1. Get all image files
        extensions = (".jpg", ".jpeg", ".png", ".webp")
        all_files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
        logger.info(f"Found {len(all_files)} images in {folder_path}")

        # 2. Filter out already imported images
        existing_files = {img.file_name for img in repo.get_all(db, limit=10000)}
        files_to_import = [f for f in all_files if f.name not in existing_files]
        
        logger.info(f"Filtered: {len(files_to_import)} new images to import.")
        if not files_to_import:
            logger.info("No new images to import.")
            return

        # 3. Process in batches
        total = len(files_to_import)
        start_time = time.time()
        
        # Prepare list of tuples (bytes, filename)
        images_to_process = []
        for f in files_to_import:
            with open(f, "rb") as rb:
                images_to_process.append((rb.read(), f.name))
        
        try:
            # Re-use the high-level service.process logic!
            # It handles batching, extraction, and saving to DB
            logger.info(f"Starting import of {total} images...")
            results = await service.process(db, images_to_process)
            db.commit()
            logger.info(f"Successfully imported {len(results)} images.")
        except Exception as e:
            db.rollback()
            logger.error(f"Import failed: {e}")

        logger.info(f"IMPORT COMPLETED! Total time: {(time.time() - start_time)/60:.2f} minutes")

    finally:
        db.close()

if __name__ == "__main__":
    # Point this to your ima_test folder
    # Assuming ima_test is at the root level c:\PTIT\2026\CSDL-PT\ima_test
    target_folder = "../ima_test" 
    asyncio.run(import_from_folder(target_folder))
