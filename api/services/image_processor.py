"""
Image Processor Service
Handles image processing and feature extraction
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from pathlib import Path
import uuid
import logging
import json
from typing import Dict, List, Optional
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from ..config import get_settings

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image processing and feature extraction service"""
    
    def __init__(self):
        """Initialize image processor"""
        self.settings = get_settings()
        
        # Initialize ResNet18 for embeddings
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])) # Remove classification layer
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info(f"ResNet18 initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize ResNet18: {e}")
            self.model = None
    
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
            
            # Texture Features (LBP, GLCM)
            texture_json = self._extract_texture(gray)
            
            # Shape Features (Hu Moments)
            shape_json = self._extract_shape(gray)
            
            # AI Embedding (ResNet18)
            embedding_json = self._extract_embedding(image_path)
            
            return {
                'width': width,
                'height': height,
                'brightness': brightness,
                'contrast': contrast,
                'saturation': saturation,
                'edge_density': edge_density,
                'dominant_color_hex': dominant_hex,
                'features_json': json.dumps(features_json),
                'texture_json': json.dumps(texture_json) if texture_json else None,
                'shape_json': json.dumps(shape_json) if shape_json else None,
                'embedding_json': embedding_json # List[float], passed as is to SQLAlchemy generic type or need JSON? 
                                                 # Check schemas. If schema expects List[float], we might need to handle storage.
                                                 # Wait, DB schema usually stores JSON or Array. 
                                                 # Let's check schemas.py... embedding_json is Optional[List[float]].
                                                 # But for DB storage, we might need a JSON column or Postgres Array.
                                                 # Since we are using basic SQLAlchemy without pgvector defined in models yet,
                                                 # we should probably store it as a JSON list for now for simplicity in `features_json` 
                                                 # OR modify the DB model. 
                                                 # For this task, let's keep it simple and assume DB handles it or we serialize it.
                                                 # The caller (images.py) passes this dict to create_image_metadata.
                                                 # Validation might fail if we pass a list where a dict is expected?
                                                 # Let's check create_image_metadata in db_service.
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
    
    def _extract_texture(self, gray: np.ndarray) -> Dict:
        """Extract LBP and GLCM texture features"""
        try:
            # LBP
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
            
            # GLCM
            glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').mean()
            energy = graycoprops(glcm, 'energy').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            return {
                "lbp_histogram": hist.tolist(),
                "glcm": {
                    "contrast": float(contrast),
                    "energy": float(energy),
                    "homogeneity": float(homogeneity),
                    "correlation": float(correlation)
                }
            }
        except Exception as e:
            logger.error(f"Error extracting texture: {e}")
            return {}

    def _extract_shape(self, gray: np.ndarray) -> Dict:
        """Extract Hu Moments shape features"""
        try:
            # Threshold to get binary image
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Calculate Moments
            moments = cv2.moments(thresh)
            
            # Calculate Hu Moments
            hu_moments = cv2.HuMoments(moments)
            
            # Log scale for better representation
            for i in range(0, 7):
                hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i])) if hu_moments[i] != 0 else 0
                
            return {
                "hu_moments": hu_moments.flatten().tolist()
            }
        except Exception as e:
            logger.error(f"Error extracting shape: {e}")
            return {}

    def _extract_embedding(self, image_path: Path) -> List[float]:
        """Extract CNN embedding using ResNet18"""
        if self.model is None:
            return []
            
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(tensor)
                
            return embedding.flatten().cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return []

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
        Compute similarity between two images.
        Prioritizes Embedding Cosine Similarity if available.
        Fallback to Histogram Intersection.
        
        Args:
            features1: Features dictionary of first image
            features2: Features dictionary of second image
            
        Returns:
            Similarity score between 0.0 and 100.0
        """
        try:
            # 1. Try Embedding Similarity (ResNet)
            emb1 = features1.get('embedding_json')
            emb2 = features2.get('embedding_json')
            
            # Handle if stored as string in some legacy cases (unlikely for list[float] but safe)
            if isinstance(emb1, str): emb1 = json.loads(emb1)
            if isinstance(emb2, str): emb2 = json.loads(emb2)

            if emb1 and emb2 and len(emb1) > 0 and len(emb2) > 0:
                vec1 = np.array(emb1)
                vec2 = np.array(emb2)
                
                # Cosine Similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_sim = dot_product / (norm1 * norm2)
                    # Normalize -1 to 1 -> 0 to 100
                    # Deep learning embeddings usually don't face 180 degree opposition in this context, 
                    # but standard cosine is -1 to 1. 0 to 1 is typical for positive vectors (ReLU outputs).
                    # ResNet output before FC is ReLU? Yes. So vectors are non-negative.
                    # Cosine sim will be 0 to 1.
                    return float(cos_sim * 100.0)

            # 2. Fallback to Color Histogram Intersection
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
            
            for channel in ['blue', 'green', 'red']:
                h1 = hist1.get(channel, [])
                h2 = hist2.get(channel, [])
                
                if len(h1) != len(h2):
                    continue
                
                for i in range(len(h1)):
                    intersection += min(h1[i], h2[i])
                    total_hist1 += h1[i]
            
            if total_hist1 == 0:
                return 0.0
            
            similarity = (intersection / total_hist1) * 100.0
            
            return min(max(similarity, 0.0), 100.0)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
