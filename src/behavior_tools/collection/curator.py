"""
Image curation module for quality filtering, deduplication, clustering, and outlier detection.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import json
import shutil

logger = logging.getLogger(__name__)


class ImageQualityFilter:
    """Filter images based on quality metrics."""

    def __init__(self, config: dict):
        """
        Initialize quality filter.

        Args:
            config: Configuration dictionary with quality parameters
        """
        self.min_resolution = config.get('min_resolution', 224)
        self.max_aspect_ratio = config.get('max_aspect_ratio', 3.0)
        self.min_aspect_ratio = config.get('min_aspect_ratio', 0.33)
        self.blur_threshold = config.get('blur_threshold', 100.0)
        self.brightness_range = config.get('brightness_range', [20, 235])

    def is_blurry(self, image_path: Path) -> bool:
        """
        Detect if image is blurry using Laplacian variance.

        Args:
            image_path: Path to image file

        Returns:
            True if blurry, False otherwise
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return True
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var < self.blur_threshold
        except Exception as e:
            logger.debug(f"Error checking blur for {image_path}: {e}")
            return True

    def check_brightness(self, image_path: Path) -> bool:
        """
        Check if image has acceptable brightness.

        Args:
            image_path: Path to image file

        Returns:
            True if brightness is acceptable, False otherwise
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            return self.brightness_range[0] <= avg_brightness <= self.brightness_range[1]
        except Exception as e:
            logger.debug(f"Error checking brightness for {image_path}: {e}")
            return False

    def check_aspect_ratio(self, image_path: Path) -> bool:
        """
        Check if image has acceptable aspect ratio.

        Args:
            image_path: Path to image file

        Returns:
            True if aspect ratio is acceptable, False otherwise
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            aspect_ratio = width / height
            return self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
        except Exception as e:
            logger.debug(f"Error checking aspect ratio for {image_path}: {e}")
            return False

    def check_resolution(self, image_path: Path) -> bool:
        """
        Check if image has minimum resolution.

        Args:
            image_path: Path to image file

        Returns:
            True if resolution is acceptable, False otherwise
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            return width >= self.min_resolution and height >= self.min_resolution
        except Exception as e:
            logger.debug(f"Error checking resolution for {image_path}: {e}")
            return False

    def filter_image(self, image_path: Path) -> bool:
        """
        Apply all quality filters to an image.

        Args:
            image_path: Path to image file

        Returns:
            True if image passes all filters, False otherwise
        """
        if not self.check_resolution(image_path):
            return False
        if not self.check_aspect_ratio(image_path):
            return False
        if not self.check_brightness(image_path):
            return False
        if self.is_blurry(image_path):
            return False
        return True


class ImageDeduplicator:
    """Remove duplicate images using perceptual hashing."""

    def __init__(self, config: dict):
        """
        Initialize deduplicator.

        Args:
            config: Configuration dictionary with deduplication parameters
        """
        self.hash_size = config.get('hash_size', 16)
        self.similarity_threshold = config.get('similarity_threshold', 5)
        self.method = config.get('method', 'phash')
        self.keep_higher_resolution = config.get('keep_higher_resolution', True)

        try:
            import imagehash
            self.imagehash = imagehash
        except ImportError:
            logger.error("imagehash not installed. Install with: pip install imagehash")
            raise

    def compute_hash(self, image_path: Path) -> str:
        """
        Compute perceptual hash of an image.

        Args:
            image_path: Path to image file

        Returns:
            Hash string
        """
        try:
            img = Image.open(image_path)

            if self.method == 'phash':
                hash_val = self.imagehash.phash(img, hash_size=self.hash_size)
            elif self.method == 'dhash':
                hash_val = self.imagehash.dhash(img, hash_size=self.hash_size)
            elif self.method == 'ahash':
                hash_val = self.imagehash.average_hash(img, hash_size=self.hash_size)
            elif self.method == 'whash':
                hash_val = self.imagehash.whash(img, hash_size=self.hash_size)
            else:
                hash_val = self.imagehash.phash(img, hash_size=self.hash_size)

            return str(hash_val)
        except Exception as e:
            logger.debug(f"Error computing hash for {image_path}: {e}")
            return None

    def find_duplicates(self, image_paths: List[Path]) -> List[Path]:
        """
        Find duplicate images in a list.

        Args:
            image_paths: List of image paths

        Returns:
            List of duplicate image paths to remove
        """
        hash_dict = {}  # hash -> (path, resolution)
        duplicates = []

        for img_path in tqdm(image_paths, desc="Finding duplicates"):
            hash_val = self.compute_hash(img_path)
            if hash_val is None:
                continue

            # Check if similar hash exists
            found_similar = False
            for existing_hash, (existing_path, existing_res) in hash_dict.items():
                # Hamming distance between hashes
                distance = sum(c1 != c2 for c1, c2 in zip(hash_val, existing_hash))

                if distance <= self.similarity_threshold:
                    found_similar = True
                    # Keep higher resolution image
                    if self.keep_higher_resolution:
                        try:
                            img = Image.open(img_path)
                            current_res = img.size[0] * img.size[1]

                            if current_res > existing_res:
                                # Replace existing with current
                                duplicates.append(existing_path)
                                hash_dict[hash_val] = (img_path, current_res)
                                del hash_dict[existing_hash]
                            else:
                                # Keep existing, mark current as duplicate
                                duplicates.append(img_path)
                        except:
                            duplicates.append(img_path)
                    else:
                        duplicates.append(img_path)
                    break

            if not found_similar:
                try:
                    img = Image.open(img_path)
                    resolution = img.size[0] * img.size[1]
                    hash_dict[hash_val] = (img_path, resolution)
                except:
                    pass

        return duplicates


class ImageClusterer:
    """Cluster images to ensure diversity."""

    def __init__(self, config: dict, model_cache_dir: str):
        """
        Initialize clusterer.

        Args:
            config: Configuration dictionary
            model_cache_dir: Directory for caching models
        """
        self.n_clusters = config.get('n_clusters', 50)
        self.samples_per_cluster = config.get('samples_per_cluster', 20)
        self.min_cluster_size = config.get('min_cluster_size', 5)
        self.method = config.get('method', 'clip')
        self.model_cache_dir = model_cache_dir

        self.model = None
        self.processor = None

    def load_model(self):
        """Load feature extraction model."""
        if self.method == 'clip':
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch

                logger.info("Loading CLIP model...")
                self.model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    cache_dir=self.model_cache_dir
                )
                self.processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    cache_dir=self.model_cache_dir
                )

                # Device selection: CUDA > MPS (Apple Silicon) > CPU
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"

                self.model.to(self.device)
                self.model.eval()
                logger.info(f"CLIP model loaded on {self.device}")
            except ImportError:
                logger.error("transformers not installed. Install with: pip install transformers")
                raise
        else:
            logger.warning(f"Method {self.method} not implemented, using CLIP")
            self.load_model()  # Default to CLIP

    def extract_features(self, image_paths: List[Path]) -> np.ndarray:
        """
        Extract features from images.

        Args:
            image_paths: List of image paths

        Returns:
            Feature array of shape (n_images, n_features)
        """
        if self.model is None:
            self.load_model()

        features = []

        import torch
        with torch.no_grad():
            for img_path in tqdm(image_paths, desc="Extracting features"):
                try:
                    image = Image.open(img_path).convert('RGB')
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    image_features = self.model.get_image_features(**inputs)
                    features.append(image_features.cpu().numpy().flatten())
                except Exception as e:
                    logger.debug(f"Error extracting features for {img_path}: {e}")
                    # Add zero vector for failed images
                    if len(features) > 0:
                        features.append(np.zeros_like(features[0]))
                    else:
                        features.append(np.zeros(512))  # CLIP feature dimension

        return np.array(features)

    def cluster_and_sample(
        self, image_paths: List[Path]
    ) -> Tuple[List[Path], Dict[int, List[Path]]]:
        """
        Cluster images and sample from each cluster.

        Args:
            image_paths: List of image paths

        Returns:
            Tuple of (selected_paths, cluster_info)
        """
        from sklearn.cluster import KMeans

        # Extract features
        features = self.extract_features(image_paths)

        # Cluster
        logger.info(f"Clustering into {self.n_clusters} clusters...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # Sample from each cluster
        cluster_info = {}
        selected_paths = []

        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]

            if len(cluster_indices) < self.min_cluster_size:
                logger.debug(f"Cluster {cluster_id} too small ({len(cluster_indices)} images), skipping")
                continue

            # Sample up to samples_per_cluster images from this cluster
            n_samples = min(self.samples_per_cluster, len(cluster_indices))
            sampled_indices = np.random.choice(cluster_indices, n_samples, replace=False)

            cluster_paths = [image_paths[i] for i in sampled_indices]
            cluster_info[cluster_id] = [str(p) for p in cluster_paths]
            selected_paths.extend(cluster_paths)

        logger.info(f"Selected {len(selected_paths)} images from {len(cluster_info)} clusters")
        return selected_paths, cluster_info


class OutlierDetector:
    """Detect and remove outlier images."""

    def __init__(self, config: dict):
        """
        Initialize outlier detector.

        Args:
            config: Configuration dictionary
        """
        self.contamination = config.get('contamination', 0.1)
        self.method = config.get('method', 'isolation_forest')

    def detect_outliers(
        self, image_paths: List[Path], features: np.ndarray
    ) -> List[Path]:
        """
        Detect outlier images.

        Args:
            image_paths: List of image paths
            features: Feature array

        Returns:
            List of outlier image paths
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor

        logger.info(f"Detecting outliers with {self.method}...")

        if self.method == 'isolation_forest':
            detector = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            predictions = detector.fit_predict(features)
        elif self.method == 'lof':
            detector = LocalOutlierFactor(
                contamination=self.contamination
            )
            predictions = detector.fit_predict(features)
        else:
            logger.warning(f"Unknown method {self.method}, using isolation_forest")
            detector = IsolationForest(contamination=self.contamination, random_state=42)
            predictions = detector.fit_predict(features)

        # -1 indicates outlier
        outlier_indices = np.where(predictions == -1)[0]
        outlier_paths = [image_paths[i] for i in outlier_indices]

        logger.info(f"Detected {len(outlier_paths)} outliers")
        return outlier_paths
