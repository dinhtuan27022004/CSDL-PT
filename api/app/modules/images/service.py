from pathlib import Path
import uuid
import json
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from .repository import ImageRepository
from .model import ImageMetadata
from .schema import ImageResponse
from ...core.config import get_settings
from ...core.logging import get_logger
from ...core.exceptions import ImageProcessingError
import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile
import torch
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import time
from skimage.feature import hog
from skimage import exposure
from typing import Tuple
logger = get_logger(__name__)

class ImageService:
    """Service for image CRUD, storage, and validation"""
    
    def __init__(self, 
        repository: ImageRepository
    ):
        self.repository = repository
        self.settings = get_settings()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load DINOv2 model and processor
        logger.info(f"Loading DINOv2 model for feature extraction on {self.device}...")
        try:
            self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base')
            self.dino_model.to(self.device)
            self.dino_model.eval()                
            logger.info("DINOv2 loaded successfully.")

            # Load CLIP model and processor
            logger.info(f"Loading CLIP model for feature extraction on {self.device}...")
            self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info(f"CLIP loaded successfully on {self.device}")
        except Exception as e:
            if self.device.type == 'cuda':
                logger.warning(f"CUDA initialization failed ({e}). Falling back to CPU.")
                self.device = torch.device('cpu')
                self.dino_model.to(self.device)
                logger.info("DINOv2 successfully fallback to CPU.")
            else:
                logger.error(f"Failed to load DINOv2 model: {e}")
                raise
    
    async def process(
        self, 
        db: Session, 
        files: List[UploadFile]
    ) -> List[ImageMetadata]:
        logger.info(f"Processing {len(files)} images")
        results = []
        start_time = time.time()
        for file in files:
            img_content = await file.read()
            result = await self.extract_features(img_content, file.filename)
            result.file_name = file.filename
            result = self.repository.create(db, result, img_content)
            results.append(result)
        logger.info(f"Processed {len(results)} images successfully after {time.time() - start_time} seconds")
        return results

    def get_images(self, db: Session, limit: int = 500, offset: int = 0) -> List[ImageMetadata]:
        return self.repository.get_all(db, limit=limit, offset=offset)

    async def extract_features(self, image: bytes, filename: Optional[str] = None) -> ImageMetadata:
        """Extract all features from image (Visual + Deep Learning) with nested extraction logic"""
        
        def _extract_brightness(img_hsv: np.ndarray) -> float:
            mean_v = np.mean(img_hsv[:, :, 2])
            return float(mean_v / 255.0)

        def _extract_contrast(img_gray: np.ndarray) -> float:
            std_dev = np.std(img_gray)
            return float(min(1.0, std_dev / 127.5))

        def _extract_saturation(img_hsv: np.ndarray) -> float:
            mean_s = np.mean(img_hsv[:, :, 1])
            return float(mean_s / 255.0)

        def _extract_edge_density(img_gray: np.ndarray) -> float:
            edges = cv2.Canny(img_gray, 100, 200)
            edge_pixels = cv2.countNonZero(edges)
            total_pixels = edges.size
            return float(edge_pixels / total_pixels)

        def _extract_histogram(img_bgr: np.ndarray) -> List[float]:
            # Convert to HSV color space - much better for color similarity
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # Each channel has 16 bins, totaling 16*3 = 48 dimensions
            # Hue (0-180), Saturation (0-255), Value (0-255)
            hist_h = cv2.calcHist([img_hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([img_hsv], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([img_hsv], [2], None, [16], [0, 256])
            
            # Combine to a single 48-dim vector
            hist_combined = np.concatenate([hist_h, hist_s, hist_v]).flatten()
            
            # Use L2 normalization
            norm = np.linalg.norm(hist_combined)
            if norm > 1e-10:
                hist_combined = hist_combined / norm
                
            return hist_combined.tolist()

        def _extract_dominant_color(img_bgr: np.ndarray) -> List[float]:
            # Resize image to small for speed while maintaining color distribution
            small_img = cv2.resize(img_bgr, (50, 50), interpolation=cv2.INTER_AREA)
            pixels = small_img.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            # K-Means clustering (K=3) to find main color groups
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Identify the most frequent cluster center
            counts = np.bincount(labels.flatten())
            dominant_bgr = centers[np.argmax(counts)]
            
            # Convert the dominant color from BGR to CIE Lab for perceptual accuracy
            dominant_bgr_img = np.uint8([[dominant_bgr]])
            dominant_lab_img = cv2.cvtColor(dominant_bgr_img, cv2.COLOR_BGR2LAB)
            
            return dominant_lab_img[0, 0].astype(float).tolist()

        def _extract_hog(img_bgr: np.ndarray, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
            # Standardize size for HOG (128x128) - Direct from original to preserve edges
            img_gray_hog = cv2.cvtColor(cv2.resize(img_bgr, (128, 128)), cv2.COLOR_BGR2GRAY)
            
            # Extract HOG with advanced L2-Hys block normalization
            fd, hog_image = hog(img_gray_hog, orientations=9, pixels_per_cell=(16, 16),
                                cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
            
            # Global L2 Normalization for better Cosine Similarity
            norm = np.linalg.norm(fd)
            if norm > 1e-10:
                fd = fd / norm
            
            vis_relative_path = None
            # Save visualization to visualizations_dir
            if filename:
                try:
                    # Improve contrast of hog_image for visualization
                    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                    vis_filename = f"hog_{filename}.png"
                    vis_path = self.settings.visualizations_dir / vis_filename
                    cv2.imwrite(str(vis_path), (hog_image_rescaled * 255).astype(np.uint8))
                    vis_relative_path = f"/static/visualizations/{vis_filename}"
                except Exception as e:
                    logger.warning(f"Could not save HOG visualization for {filename}: {e}")
                
            return fd.tolist(), vis_relative_path

        def _extract_hu_moments(img_bgr: np.ndarray, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
            # Standardize size for HU (256x256)
            img_input = cv2.resize(img_bgr, (256, 256))
            img_gray_hu = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            
            # Step 1: Smooth the image to reduce noise
            blurred = cv2.GaussianBlur(img_gray_hu, (5, 5), 0)
            
            # Step 2: Canny Edge Detection is better for defining silhouettes than simple thresholding
            edged = cv2.Canny(blurred, 50, 150)
            
            # Step 3: Dilate edges to close gaps
            dilated = cv2.dilate(edged, None, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            moments = None
            vis_relative_path = None
            
            if contours:
                # FILTER: Get rid of noise (too small) and the image frame (too large)
                valid_contours = []
                for c in contours:
                    area = cv2.contourArea(c)
                    # Filter out very small noise and anything taking more than 90% of the image (likely the frame)
                    if 100 < area < (256 * 256 * 0.9):
                        valid_contours.append(c)
                
                if valid_contours:
                    # Select the largest valid contour
                    cnt = max(valid_contours, key=cv2.contourArea)
                    moments = cv2.moments(cnt)
                    
                    # Visualization: Draw green contour and red centroid
                    if filename:
                        try:
                            vis_hu = np.zeros((256, 256, 3), dtype=np.uint8)
                            cv2.drawContours(vis_hu, [cnt], -1, (0, 255, 0), 2)
                            
                            if moments['m00'] > 0:
                                cX = int(moments["m10"] / moments["m00"])
                                cY = int(moments["m01"] / moments["m00"])
                                cv2.drawMarker(vis_hu, (cX, cY), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                            
                            vis_filename = f"hu_{filename}.png"
                            vis_path = self.settings.visualizations_dir / vis_filename
                            cv2.imwrite(str(vis_path), vis_hu)
                            vis_relative_path = f"/static/visualizations/{vis_filename}"
                        except Exception as e:
                            logger.warning(f"HU visualization failed: {e}")
            
            # Fallback if no valid object contour found
            if moments is None:
                _, thresh = cv2.threshold(img_gray_hu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                moments = cv2.moments(thresh)
                if filename and not vis_relative_path:
                    try:
                        vis_hu = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                        vis_filename = f"hu_{filename}.png"
                        vis_path = self.settings.visualizations_dir / vis_filename
                        cv2.imwrite(str(vis_path), vis_hu)
                        vis_relative_path = f"/static/visualizations/{vis_filename}"
                    except Exception as e:
                        logger.warning(f"HU visualization failed: {e}")

            # Log transform Hu Moments
            hu = cv2.HuMoments(moments).flatten()
            hu_transformed = []
            for h in hu:
                if abs(h) > 1e-20:
                    hu_transformed.append(-1.0 * np.sign(h) * np.log10(abs(h)))
                else:
                    hu_transformed.append(0.0)
            return hu_transformed, vis_relative_path
        
        def _extract_dinov2_features(img_bgr: np.ndarray) -> List[float]:
            rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            inputs = self.dino_processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            vector = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            vector = vector / (np.linalg.norm(vector) + 1e-10)
            return vector.tolist()

        def _extract_clip_features(img_bgr: np.ndarray) -> List[float]:
            rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # CLIP get_image_features usually returns a direct tensor
                outputs = self.clip_model.get_image_features(**inputs)
            
            # Safe extraction for CLIP
            if torch.is_tensor(outputs):
                vector = outputs.cpu().numpy().flatten()
            elif hasattr(outputs, 'image_embeds'):
                vector = outputs.image_embeds.cpu().numpy().flatten()
            else:
                # If it's a generic output object, try pooler_output
                vector = getattr(outputs, 'pooler_output', outputs[0]).cpu().numpy().flatten()
                
            vector = vector / (np.linalg.norm(vector) + 1e-10)
            return vector.tolist()

        try:
            file_bytes = np.frombuffer(image, dtype=np.uint8)
            img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_original is None:
                raise ValueError("Failed to decode image")
            
            # Use smaller version for traditional features to speed up
            img_small = cv2.resize(img_original, (int(img_original.shape[1]/6), int(img_original.shape[0]/6)))
            img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
            
            brightness = _extract_brightness(img_hsv)
            contrast = _extract_contrast(img_gray)
            saturation = _extract_saturation(img_hsv)
            edge_density = _extract_edge_density(img_gray)
            
            histogram_vector = _extract_histogram(img_small)
            hog_vector, hog_vis_path = _extract_hog(img_original, filename)
            hu_moments_vector, hu_vis_path = _extract_hu_moments(img_original, filename)
            
            return ImageMetadata(
                width = img_original.shape[1],
                height = img_original.shape[0],
                brightness = brightness,
                contrast = contrast,
                saturation = saturation,
                edge_density = edge_density,
                histogram_vector = histogram_vector,
                hog_vector = hog_vector,
                hog_vis_path = hog_vis_path,
                hu_moments_vector = hu_moments_vector,
                hu_vis_path = hu_vis_path,
                dinov2_vector = _extract_dinov2_features(img_original),
                clip_vector = _extract_clip_features(img_original),
                dominant_color_vector = _extract_dominant_color(img_original)
            )
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise

    async def search_similar(
        self, 
        db: Session, 
        query_image_content: bytes, 
        filename: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Hybrid Search with 5-way weighted similarity mapping"""
        try:
            query_metadata = await self.extract_features(query_image_content, filename)
            
            # Use Repository for high-performance 5-way search
            search_results = self.repository.search(
                db=db, 
                query_metadata=query_metadata,
                limit=limit
            )
            
            # Map database results (tuples) to ImageResponse schema objects
            response_data = []
            for row in search_results:
                # row structure: (record, similarity, dino_sim, clip_sim, dom_sim, b_sim, c_sim, s_sim, e_sim, h_sim, hog_sim, hu_sim)
                record, total_sim, dino_sim, clip_sim, dom_sim, b_sim, c_sim, s_sim, e_sim, h_sim, hog_sim, hu_sim = row
                
                # Convert 0-1 similarity to 0-100% for the UI
                res = ImageResponse.model_validate(record)
                res.similarity = round(float(total_sim) * 100.0, 2)
                res.dino_similarity = round(float(dino_sim) * 100.0, 2)
                res.clip_similarity = round(float(clip_sim) * 100.0, 2)
                res.dominant_color_similarity = round(float(dom_sim) * 100.0, 2)
                res.brightness_similarity = round(float(b_sim) * 100.0, 2)
                res.contrast_similarity = round(float(c_sim) * 100.0, 2)
                res.saturation_similarity = round(float(s_sim) * 100.0, 2)
                res.edge_density_similarity = round(float(e_sim) * 100.0, 2)
                res.histogram_similarity = round(float(h_sim) * 100.0, 2)
                res.hog_similarity = round(float(hog_sim) * 100.0, 2)
                res.hu_moments_similarity = round(float(hu_sim) * 100.0, 2)
                
                response_data.append(res)
                
            return {
                "query_features": {
                    "width": query_metadata.width,
                    "height": query_metadata.height,
                    "brightness": query_metadata.brightness,
                    "contrast": query_metadata.contrast,
                    "saturation": query_metadata.saturation,
                    "edge_density": query_metadata.edge_density,
                    "histogram_vector": query_metadata.histogram_vector,
                    "dinov2_vector": query_metadata.dinov2_vector
                },
                "results": response_data
            }
        except Exception as e:
            logger.error(f"Error in search_similar: {e}")
            raise
