import os
# Force offline mode for transformers/hf_hub at the very beginning
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import base64
import httpx
import logging
import gc
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from ..core.config import get_settings
from .cache_service import CacheService

logger = logging.getLogger(__name__)
settings = get_settings()

class LLMService:
    """Service for interacting with OpenRouter LLM and Embedding APIs with JSON caching"""
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.api_key = settings.openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache = cache_service or CacheService()
        self.model = None # Loaded dynamically

    def load_embedding_model(self):
        """Loads local BGE-M3 Model into GPU memory"""
        if self.model is not None:
            return
        
        logger.info(f"Loading local embedding model: {settings.llm_embedding_model} on {self.device}...")
        try:
            # Try loading with local_files_only first
            self.model = SentenceTransformer(
                settings.llm_embedding_model, 
                device=self.device, 
                local_files_only=True,
                trust_remote_code=False
            )
            logger.info(f"Model loaded OFFLINE on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load model locally ({e}). Attempting normal load...")
            try:
                # If local_files_only fails, try normal load (might still work if cached but offline flag was too strict)
                self.model = SentenceTransformer(
                    settings.llm_embedding_model, 
                    device=self.device,
                    trust_remote_code=False
                )
                logger.info(f"Model loaded on {self.device}")
            except Exception as final_e:
                logger.error(f"CRITICAL: Could not load embedding model: {final_e}")
                self.model = None
                raise

    def unload_embedding_model(self):
        """Unloads BGE-M3 Model and frees GPU memory"""
        if self.model is not None:
            logger.info("Unloading embedding model...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Embedding model unloaded and GPU cache cleared")

    async def analyze_vision_batch(self, images_bytes: List[bytes], filenames: List[str]) -> List[Dict[str, Any]]:
        """Analyzes a batch of images using LLM Vision API concurrently"""
        semaphore = asyncio.Semaphore(50)  # Limit to 50 concurrent requests to prevent file descriptor limit
        
        async def bounded_analyze(img, name):
            async with semaphore:
                return await self._analyze_single_vision(img, name)
                
        tasks = []
        for img, name in zip(images_bytes, filenames):
            tasks.append(bounded_analyze(img, name))
        return await asyncio.gather(*tasks)

    async def _analyze_single_vision(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Single vision analysis with caching check"""
        cached_item = self.cache.get_item(filename)
        if cached_item and "description" in cached_item and "category" in cached_item:
            logger.info(f"Using cached vision data for: {filename}")
            return {
                "category": cached_item.get("category"),
                "description": cached_item.get("description"),
                "entities": cached_item.get("entities", []),
                "cached": True
            }

        if not self.api_key:
            return {"category": "Unknown", "description": "No API key", "entities": [], "cached": False}

        try:
            analysis = await self._call_vision_api(image_bytes)
            self.cache.update_item(filename, analysis)
            return {**analysis, "cached": False}
        except Exception as e:
            logger.error(f"Vision API failed for {filename}: {e}")
            return {"category": "Error", "description": str(e), "entities": [], "cached": False}

    def extract_embeddings_batch(self, texts: List[str], filenames: Optional[List[str]] = None, batch_size: int = 32) -> List[Optional[List[float]]]:
        """Extracts embeddings for a list of texts using sequential GPU management"""
        if not texts:
            return []
        
        self.load_embedding_model()
        try:
            embeddings = []
            valid_texts = [t for t in texts if t]
            if not valid_texts:
                return [None] * len(texts)
            
            logger.info(f"Encoding {len(valid_texts)} texts into embeddings with batch_size={batch_size}...")
            encoded = self.model.encode(valid_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
            logger.info(f"Finished encoding {len(valid_texts)} texts.")
            
            res_idx = 0
            for t in texts:
                if t:
                    embeddings.append(encoded[res_idx].tolist())
                    res_idx += 1
                else:
                    embeddings.append(None)
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [None] * len(texts)
        finally:
            self.unload_embedding_model()

    async def analyze_and_embed(
        self, 
        image_bytes: bytes, 
        filename: str, 
        force: bool = False
    ) -> Dict[str, Any]:
        """Backward compatibility for single image analysis and embedding"""
        vision_res = await self._analyze_single_vision(image_bytes, filename)
        
        description = vision_res.get("description", "")
        category = vision_res.get("category", "General")
        entities = vision_res.get("entities", [])
        
        embedding = None
        if description:
            combined_text = f"Category: {category}. Entities: {entities} . Description: {description}"
            self.load_embedding_model()
            embedding = self._call_embedding_api(combined_text)
            self.unload_embedding_model()
            
        return {
            "category": category,
            "description": description,
            "entities": entities,
            "llm_embedding": embedding
        }

    async def _call_vision_api(self, image_bytes: bytes) -> Dict[str, str]:
        """Calls OpenRouter Vision model"""
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:5173",
            "Content-Type": "application/json"
        }
        
        prompt = (
            "Analyze this image and provide: "
            "1. A short category (e.g., Nature, Technology, Architecture, People, Vehicle). "
            "2. A detailed description in Vietnamese focusing on visual elements, colors, and mood. "
            "3. A list of specific English entity names detected (e.g., ['car', 'tree', 'sunset']). "
            "Output strictly as JSON: {\"category\": \"string\", \"description\": \"string\", \"entities\": [\"string\"]}"
        )
        
        payload = {
            "model": settings.llm_vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data['choices'][0]['message']['content']
            return json.loads(content)

    def _call_embedding_api(self, text: str) -> Optional[List[float]]:
        """Internal helper for single embedding (expects model to be loaded)"""
        try:
            if not self.model:
                return None
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            return None
