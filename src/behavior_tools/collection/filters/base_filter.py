"""
Base filter interface for multi-model filtering framework.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseFilter(ABC):
    """Abstract base class for image filters."""

    def __init__(self, config: dict, model_cache_dir: Optional[str] = None):
        """
        Initialize filter.

        Args:
            config: Filter configuration
            model_cache_dir: Directory for caching models
        """
        self.config = config
        self.model_cache_dir = model_cache_dir or "./models"
        self.model = None
        self.device = self._get_device(config.get('device', 'auto'))

    def _get_device(self, device_config: str) -> str:
        """Determine device (auto, cuda, mps, cpu)."""
        if device_config != 'auto':
            return device_config

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    @abstractmethod
    def load_model(self):
        """Load the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def filter_image(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Filter a single image.

        Args:
            image_path: Path to image

        Returns:
            Tuple of (is_accepted, details_dict)
        """
        pass

    def filter_batch(
        self,
        image_paths: List[Path],
        show_progress: bool = True
    ) -> Tuple[List[Path], List[Path], Dict]:
        """
        Filter a batch of images.

        Args:
            image_paths: List of image paths
            show_progress: Show progress bar

        Returns:
            Tuple of (accepted_paths, rejected_paths, stats)
        """
        if self.model is None:
            self.load_model()

        accepted = []
        rejected = []
        details_list = []

        iterator = image_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc=f"Filtering ({self.__class__.__name__})")
            except ImportError:
                pass

        for img_path in iterator:
            is_accepted, details = self.filter_image(img_path)

            if is_accepted:
                accepted.append(img_path)
            else:
                rejected.append(img_path)

            details_list.append(details)

        # Compute statistics
        stats = {
            'filter_type': self.__class__.__name__,
            'total': len(image_paths),
            'accepted': len(accepted),
            'rejected': len(rejected),
            'acceptance_rate': len(accepted) / len(image_paths) if image_paths else 0,
            'details': details_list
        }

        # Count rejection reasons
        rejection_reasons = {}
        for d in details_list:
            if not d.get('accepted', False):
                reason = d.get('reason', 'unknown')
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        stats['rejection_reasons'] = rejection_reasons

        return accepted, rejected, stats

    def __str__(self):
        return f"{self.__class__.__name__}(device={self.device})"
