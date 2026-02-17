"""
Image Processor Service
Handles image processing and feature extraction
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import uuid
import logging
import json
from typing import Dict

from ..config import get_settings

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image processing and feature extraction service"""
    
    def __init__(self):
        """Initialize image processor"""
        self.settings = get_settings()
    
    def save_upload(self, content: bytes, original_filename: str) -> tuple[Path, str]:

        try:
            # Generate unique filename
            file_ext = Path(original_filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = self.settings.uploads_dir / unique_filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Saved file: {file_path}")
            return file_path, unique_filename
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    def extract_features(self, image_path: Path) -> Dict:

        try:
            with open(image_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            print(f"✅ img: {img}")
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            height, width = img.shape[:2]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Brightness (mean of grayscale)
            brightness = float(np.mean(gray) / 255.0)
            
            # Contrast (std of grayscale)
            contrast = float(np.std(gray) / 255.0)
            
            # Saturation (mean of S channel in HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
            
            # Edge density (Canny edges)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = float(np.sum(edges > 0) / (height * width))
            
            # Dominant color (k-means on downsampled pixels)
            dominant_hex = self._extract_dominant_color(img)
            
            # Histogram for features_json
            features_json = self._extract_histogram_features(img, edge_density, contrast)
            
            return {
                'width': width,
                'height': height,
                'brightness': brightness,
                'contrast': contrast,
                'saturation': saturation,
                'edge_density': edge_density,
                'dominant_color_hex': dominant_hex,
                'features_json': json.dumps(features_json)
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _extract_dominant_color(self, img: np.ndarray) -> str:
        """
        Extract dominant color using K-means clustering
        
        Args:
            img: OpenCV image (BGR format)
            
        Returns:
            Hex color string
        """
        # Downsample for performance
        small_img = cv2.resize(img, (100, 100))
        pixels = small_img.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=1, n_init=1, random_state=42)
        kmeans.fit(pixels)
        dominant_bgr = kmeans.cluster_centers_[0]
        
        # Convert BGR to RGB
        dominant_rgb = dominant_bgr[::-1]
        
        # Convert to hex
        dominant_hex = '#{:02x}{:02x}{:02x}'.format(
            int(dominant_rgb[0]),
            int(dominant_rgb[1]),
            int(dominant_rgb[2])
        )
        
        return dominant_hex
    
    def _extract_histogram_features(
        self,
        img: np.ndarray,
        edge_density: float,
        contrast: float
    ) -> Dict:
        """
        Extract histogram and additional features
        
        Args:
            img: OpenCV image (BGR format)
            edge_density: Pre-computed edge density
            contrast: Pre-computed contrast
            
        Returns:
            Dictionary of histogram features
        """
        hist_b = cv2.calcHist([img], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [16], [0, 256])
        
        return {
            "histogram": {
                "blue": hist_b.flatten().tolist(),
                "green": hist_g.flatten().tolist(),
                "red": hist_r.flatten().tolist()
            },
            "texture_score": float(edge_density),
            "quality_score": float(contrast)
        }
    
    def validate_image(self, content_type: str) -> bool:
        """
        Validate if file is an image
        
        Args:
            content_type: MIME content type
            
        Returns:
            True if valid image
        """
        return content_type.startswith('image/')

    def compute_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Compute similarity between two images based on their features.
        Uses Histogram Intersection on color histograms.
        
        Args:
            features1: Features dictionary of first image
            features2: Features dictionary of second image
            
        Returns:
            Similarity score between 0.0 and 100.0
        """
        try:
            # Parse features_json if it's a string
            f1_json = features1.get('features_json')
            f2_json = features2.get('features_json')

            if isinstance(f1_json, str):
                f1_json = json.loads(f1_json)
            if isinstance(f2_json, str):
                f2_json = json.loads(f2_json)

            if not f1_json or not f2_json:
                return 0.0

            hist1 = f1_json.get('histogram', {})
            hist2 = f2_json.get('histogram', {})

            # Calculate histogram intersection
            intersection = 0.0
            total_hist1 = 0.0
            
            # Sum for normalization (assuming histograms are normalized to total pixels generally, 
            # but let's be safe and sum them up)
            
            for channel in ['blue', 'green', 'red']:
                h1 = hist1.get(channel, [])
                h2 = hist2.get(channel, [])
                
                if len(h1) != len(h2):
                    continue
                
                # Intersection kernel
                for i in range(len(h1)):
                    intersection += min(h1[i], h2[i])
                    total_hist1 += h1[i]
            
            # Avoid division by zero
            if total_hist1 == 0:
                return 0.0
            
            # Normalize to 0-1 range then convert to percentage
            similarity = (intersection / total_hist1) * 100.0
            
            return min(max(similarity, 0.0), 100.0)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
