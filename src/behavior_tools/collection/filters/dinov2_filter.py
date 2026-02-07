"""
DINOv2-based visual similarity filter using reference images.
"""
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image
import logging

from .base_filter import BaseFilter

logger = logging.getLogger(__name__)


class DINOv2Filter(BaseFilter):
    """DINOv2 model for reference-based similarity filtering."""

    def __init__(self, config: dict, model_cache_dir: str = None):
        super().__init__(config, model_cache_dir)
        self.positive_features = None
        self.negative_features = None
        self.positive_paths = []
        self.negative_paths = []

    def load_model(self):
        """Load DINOv2 model."""
        try:
            from transformers import AutoImageProcessor, AutoModel
            import torch

            logger.info(f"Loading DINOv2 model on {self.device}...")

            model_name = self.config.get('model_name', 'facebook/dinov2-base')

            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            )

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"✓ DINOv2 loaded on {self.device}")

            # Load reference images
            self._load_reference_images()

        except ImportError as e:
            logger.error(f"Failed to load DINOv2: {e}")
            logger.error("Install with: pip install transformers torch")
            raise

    def _load_reference_images(self):
        """Load and process positive/negative reference images."""
        import torch

        target_config = self.config.get('target', {})
        reference_dir = Path(target_config.get('reference_dir', 'reference_images'))

        if not reference_dir.exists():
            logger.warning(f"Reference directory not found: {reference_dir}")
            logger.warning("Filter will not work properly!")
            return

        # Load positive references
        positive_dir = reference_dir / "positive"
        if positive_dir.exists():
            self._load_reference_set(positive_dir, is_positive=True)
        else:
            # Fallback: use all images in reference_dir as positive
            logger.info(f"No 'positive' subfolder. Using all images in {reference_dir}")
            self._load_reference_set(reference_dir, is_positive=True)

        # Load negative references (optional)
        negative_dir = reference_dir / "negative"
        if negative_dir.exists():
            self._load_reference_set(negative_dir, is_positive=False)
        else:
            logger.info("No 'negative' subfolder. Negative filtering disabled.")

    def _load_reference_set(self, directory: Path, is_positive: bool):
        """Load images from a directory."""
        ref_type = "positive" if is_positive else "negative"

        # Find all images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(directory.glob(ext))

        if not images:
            logger.warning(f"No {ref_type} images found in {directory}")
            return

        logger.info(f"Loading {len(images)} {ref_type} reference images from {directory}")

        # Extract features
        features = []
        paths = []

        for img_path in images:
            try:
                feature = self._extract_features(img_path)
                if feature is not None:
                    features.append(feature)
                    paths.append(img_path)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")

        if features:
            if is_positive:
                self.positive_features = np.array(features)
                self.positive_paths = paths
                logger.info(f"✓ Loaded {len(features)} positive features")
            else:
                self.negative_features = np.array(features)
                self.negative_paths = paths
                logger.info(f"✓ Loaded {len(features)} negative features")
        else:
            logger.warning(f"No valid {ref_type} features extracted!")

    def _extract_features(self, image_path: Path) -> np.ndarray:
        """Extract DINOv2 features from an image."""
        import torch

        try:
            image = Image.open(image_path).convert('RGB')

            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token (first token) as image representation
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return features.squeeze()

        except Exception as e:
            logger.debug(f"Error extracting features from {image_path}: {e}")
            return None

    def _compute_similarity(self, features: np.ndarray) -> tuple:
        """
        Compute similarity with positive and negative references.

        Returns:
            (positive_similarity, negative_similarity)
        """
        # Normalize features
        features_norm = features / (np.linalg.norm(features) + 1e-8)

        # Positive similarity
        positive_sim = 0.0
        if self.positive_features is not None and len(self.positive_features) > 0:
            positive_norm = self.positive_features / (np.linalg.norm(self.positive_features, axis=1, keepdims=True) + 1e-8)
            positive_similarities = np.dot(positive_norm, features_norm)
            positive_sim = float(np.max(positive_similarities))
        else:
            logger.warning("No positive reference features!")

        # Negative similarity
        negative_sim = 0.0
        if self.negative_features is not None and len(self.negative_features) > 0:
            negative_norm = self.negative_features / (np.linalg.norm(self.negative_features, axis=1, keepdims=True) + 1e-8)
            negative_similarities = np.dot(negative_norm, features_norm)
            negative_sim = float(np.max(negative_similarities))

        return positive_sim, negative_sim

    def filter_image(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Filter image using DINOv2 similarity to reference images.

        Returns:
            (is_accepted, details)
        """
        if self.model is None:
            self.load_model()

        try:
            # Extract features
            features = self._extract_features(image_path)

            if features is None:
                return False, {
                    'image_path': str(image_path),
                    'reason': 'feature_extraction_failed',
                    'accepted': False
                }

            # Compute similarity
            positive_sim, negative_sim = self._compute_similarity(features)

            # Get thresholds
            similarity_threshold = self.config.get('similarity_threshold', 0.70)
            uncertain_range = self.config.get('uncertain_range', [0.50, 0.70])

            # Build details
            details = {
                'image_path': str(image_path),
                'positive_similarity': positive_sim,
                'negative_similarity': negative_sim,
                'accepted': False,
                'reason': '',
                'uncertain': False
            }

            # Decision logic with positive/negative
            # 1. Reject if too similar to negative examples
            if negative_sim > 0.60:
                details['reason'] = 'similar_to_negative'
                return False, details

            # 2. Accept if highly similar to positive AND not similar to negative
            if positive_sim >= similarity_threshold and negative_sim < 0.50:
                details['accepted'] = True
                details['reason'] = 'high_positive_similarity'
                return True, details

            # 3. Uncertain range → flag for manual review
            elif positive_sim >= uncertain_range[0] and negative_sim < 0.60:
                details['reason'] = 'uncertain_similarity'
                details['uncertain'] = True
                return False, details

            # 4. Low similarity → Reject
            else:
                details['reason'] = 'low_positive_similarity'
                return False, details

        except Exception as e:
            logger.error(f"Error filtering {image_path}: {e}")
            return False, {
                'image_path': str(image_path),
                'reason': 'error',
                'error': str(e),
                'accepted': False
            }
