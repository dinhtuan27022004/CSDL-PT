import json
import logging
from typing import Dict, Any, Optional
from ..core.config import get_settings
import threading

logger = logging.getLogger(__name__)
settings = get_settings()

class CacheService:
    """Centralized service for managing image feature and semantic cache"""
    
    _lock = threading.Lock() # Class-level lock for file operations
    
    def __init__(self, cache_filename: str = "semantic_cache.json"):
        # Store cache in the services directory or a dedicated data directory
        self.cache_path = settings.base_dir / "app" / "services" / cache_filename
        self._ensure_cache_exists()

    def _ensure_cache_exists(self):
        if not self.cache_path.exists():
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            except Exception as e:
                logger.error(f"Failed to create cache file: {e}")

    def _load_cache(self) -> Dict[str, Any]:
        try:
            if not self.cache_path.exists():
                return {}
            
            # Check if file is empty
            if self.cache_path.stat().st_size == 0:
                logger.warning(f"Cache file {self.cache_path.name} is empty. Initializing new cache.")
                return {}

            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Cache file {self.cache_path.name} is corrupted. Returning empty cache.")
            return {}
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return {}

    def _save_cache(self, cache: Dict[str, Any]):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_item(self, filename: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data for a specific image filename"""
        if not filename:
            return None
        with self._lock:
            cache = self._load_cache()
            return cache.get(filename)

    def update_item(self, filename: str, data: Dict[str, Any]):
        """Merge new data into a cached item or create a new one"""
        if not filename:
            return
        with self._lock:
            cache = self._load_cache()
            if filename not in cache:
                cache[filename] = {}
            
            # Consistent merging
            cache[filename].update(data)
            self._save_cache(cache)

    def delete_item(self, filename: str):
        """Remove an item from the cache"""
        with self._lock:
            cache = self._load_cache()
            if filename in cache:
                del cache[filename]
                self._save_cache(cache)
