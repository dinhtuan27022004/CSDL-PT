import os
import json
import base64
import httpx
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class LLMService:
    """Service for interacting with OpenRouter LLM and Embedding APIs with JSON caching"""
    
    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Local BGE-M3 Model
        logger.info(f"Loading local embedding model: {settings.llm_embedding_model}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(settings.llm_embedding_model, device=device)
        logger.info(f"Model loaded on {device}")
        
        self.cache_path = settings.base_dir / "app" / "modules" / "images" / "semantic_cache.json"
        self._ensure_cache_exists()

    def _ensure_cache_exists(self):
        if not self.cache_path.exists():
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_cache(self) -> Dict[str, Any]:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load LLM cache: {e}")
            return {}

    def _save_cache(self, cache: Dict[str, Any]):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save LLM cache: {e}")

    async def analyze_and_embed(
        self, 
        image_bytes: bytes, 
        filename: str, 
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze image for category/description and get embeddings.
        Uses local JSON cache to prevent redundant API calls.
        """
        cache = self._load_cache()
        
        if not force and filename in cache:
            logger.info(f"Using cached semantic data for: {filename}")
            cached_item = cache[filename]
            
            # Tái tạo embedding từ text đã cache để tránh lưu vector nặng vào JSON
            category = cached_item.get("category", "General").lower()
            description = cached_item.get("description", "")
            entities = [e.lower() for e in cached_item.get("entities", [])]
            
            embedding = None
            if description and category and entities:
                combined_text = f"Category: {category}. Entities: {entities} . Description: {description}"
                embedding = self._call_embedding_api(combined_text)
                
                return {
                    **cached_item,
                    "llm_embedding": embedding
                }

        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set. Skipping semantic analysis.")
            return {"category": "Unknown", "description": "Analysis skipped (no API key)", "llm_embedding": None}

        logger.info(f"Calling OpenRouter for semantic analysis: {filename}")
        
        try:
            # 1. Image Analysis (Vision)
            analysis = await self._call_vision_api(image_bytes)
            category = analysis.get("category", "General").lower()
            description = analysis.get("description", "")
            entities = [e.lower() for e in analysis.get("entities", [])]
            
            # 2. Local Text Embedding (BGE-M3)
            embedding = None
            if description:
                # Kết hợp Category vào Description để tăng cường độ chính xác khi tìm kiếm ngữ nghĩa
                combined_text = f"Category: {category}. Entities: {entities} . Description: {description}"
                embedding = self._call_embedding_api(combined_text)
            
            result = {
                "category": category,
                "description": description,
                "entities": entities,
                "llm_embedding": embedding
            }
            
            # 3. Update Cache (Chỉ lưu text, không lưu vector để giảm dung lượng file JSON)
            cache_data = result.copy()
            if "llm_embedding" in cache_data:
                del cache_data["llm_embedding"]
                
            cache[filename] = cache_data
            self._save_cache(cache)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM analysis failed for {filename}: {e}")
            return {"category": "Error", "description": str(e), "llm_embedding": None}

    async def _call_vision_api(self, image_bytes: bytes) -> Dict[str, str]:
        """Calls OpenRouter Vision model"""
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:5173", # Optional, for OpenRouter analytics
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
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            content = data['choices'][0]['message']['content']
            return json.loads(content)

    def _call_embedding_api(self, text: str) -> Optional[List[float]]:
        """Uses local BGE-M3 model to generate embeddings"""
        try:
            # BGE-M3 can benefit from pooling (default is often fine)
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            return None
