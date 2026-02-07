"""
CLIP-based semantic filter with improved prompts.
"""
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
from PIL import Image
import logging

from .base_filter import BaseFilter

logger = logging.getLogger(__name__)


class CLIPFilter(BaseFilter):
    """CLIP model for semantic filtering."""

    def load_model(self):
        """Load CLIP model."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch

            logger.info(f"Loading CLIP model on {self.device}...")

            model_name = self.config.get('model_name', 'openai/clip-vit-base-patch32')

            self.model = CLIPModel.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            )
            self.processor = CLIPProcessor.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            )

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"✓ CLIP loaded on {self.device}")

        except ImportError as e:
            logger.error(f"Failed to load CLIP: {e}")
            logger.error("Install with: pip install transformers torch")
            raise

    def _compute_similarity(self, image: Image.Image, text_prompts: list) -> float:
        """Compute max similarity between image and text prompts."""
        import torch

        try:
            # Process inputs
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            # Get features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                text_features = self.model.get_text_features(**text_inputs)

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Cosine similarity
                similarity = (image_features @ text_features.T).cpu().numpy()

            return float(np.max(similarity))

        except Exception as e:
            logger.debug(f"Error computing similarity: {e}")
            return 0.0

    def filter_image(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Filter image using CLIP.

        Returns:
            (is_accepted, details)
        """
        if self.model is None:
            self.load_model()

        try:
            image = Image.open(image_path).convert('RGB')

            # Get prompts from config
            target_config = self.config.get('target', {})
            positive_concepts = target_config.get('positive_concepts', [])
            negative_concepts_dict = target_config.get('negative_concepts', {})

            # Compute scores
            target_score = self._compute_similarity(image, positive_concepts)

            # Check negative concepts
            negative_scores = {}
            reject_reason = None

            for category, neg_config in negative_concepts_dict.items():
                if isinstance(neg_config, dict):
                    prompts = neg_config.get('keywords', neg_config.get('prompts', []))
                    threshold = neg_config.get('threshold', 0.30)
                else:
                    continue

                if prompts:
                    score = self._compute_similarity(image, prompts)
                    negative_scores[category] = score

                    if score > threshold:
                        reject_reason = f"negative_{category}"
                        break

            # Build details
            details = {
                'image_path': str(image_path),
                'target_score': target_score,
                'negative_scores': negative_scores,
                'accepted': False,
                'reason': ''
            }

            # Decision logic
            min_target_score = self.config.get('min_target_score', 0.40)

            # Reject if negative match
            if reject_reason:
                details['reason'] = reject_reason
                return False, details

            # Reject if low target score
            if target_score < min_target_score:
                details['reason'] = 'low_target_score'
                return False, details

            # Accept
            details['accepted'] = True
            details['reason'] = 'passed'
            return True, details

        except Exception as e:
            logger.error(f"Error filtering {image_path}: {e}")
            return False, {
                'image_path': str(image_path),
                'reason': 'error',
                'error': str(e),
                'accepted': False
            }
