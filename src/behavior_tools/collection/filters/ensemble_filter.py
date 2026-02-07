"""
Ensemble filter combining multiple models.
"""
from pathlib import Path
from typing import List, Tuple, Dict
import logging

from .base_filter import BaseFilter

logger = logging.getLogger(__name__)


class EnsembleFilter(BaseFilter):
    """Combine multiple filters with weighted voting."""

    def __init__(self, filters: List[BaseFilter], config: dict):
        """
        Initialize ensemble.

        Args:
            filters: List of filter instances
            config: Ensemble configuration
        """
        self.filters = filters
        self.config = config
        self.weights = config.get('weights', {})
        self.device = filters[0].device if filters else 'cpu'
        self.model = None  # Ensemble doesn't have its own model

    def load_model(self):
        """Load all models."""
        for f in self.filters:
            if hasattr(f, 'model') and f.model is None:
                f.load_model()

    def filter_batch(self, image_paths: List[Path], show_progress: bool = False) -> Tuple[List[Path], List[Path], Dict]:
        """
        Override filter_batch to handle ensemble-specific logic.

        Args:
            image_paths: List of image paths to filter
            show_progress: Whether to show progress bar

        Returns:
            (accepted_paths, rejected_paths, stats)
        """
        # Load all models first
        self.load_model()

        # Use parent's filter_batch implementation
        from tqdm import tqdm

        accepted = []
        rejected = []
        details_list = []

        iterator = tqdm(image_paths, desc="Filtering") if show_progress else image_paths

        for img_path in iterator:
            is_accepted, details = self.filter_image(img_path)

            if is_accepted:
                accepted.append(img_path)
            else:
                rejected.append(img_path)

            details_list.append(details)

        stats = {
            'total': len(image_paths),
            'accepted': len(accepted),
            'rejected': len(rejected),
            'details': details_list
        }

        return accepted, rejected, stats

    def filter_image(self, image_path: Path) -> Tuple[bool, Dict]:
        """
        Filter using ensemble of models.

        Returns:
            (is_accepted, details)
        """
        if any(f.model is None for f in self.filters):
            self.load_model()

        # Get predictions from all filters
        predictions = []
        all_details = {}

        for f in self.filters:
            is_accepted, details = f.filter_image(image_path)
            predictions.append({
                'filter': f.__class__.__name__,
                'accepted': is_accepted,
                'details': details
            })
            all_details[f.__class__.__name__] = details

        # Weighted voting
        total_score = 0.0
        total_weight = 0.0

        for pred in predictions:
            filter_name = pred['filter']
            weight = self.weights.get(filter_name.lower().replace('filter', ''), 0.5)

            # Score: 1.0 if accepted, 0.0 if rejected
            score = 1.0 if pred['accepted'] else 0.0

            # Special handling for uncertain cases
            if pred['details'].get('uncertain', False):
                score = 0.5  # Neutral score

            total_score += score * weight
            total_weight += weight

        # Final decision
        ensemble_score = total_score / total_weight if total_weight > 0 else 0.0
        is_accepted = ensemble_score >= 0.5

        # Build combined details
        details = {
            'image_path': str(image_path),
            'ensemble_score': ensemble_score,
            'accepted': is_accepted,
            'reason': 'ensemble_accepted' if is_accepted else 'ensemble_rejected',
            'individual_predictions': predictions,
            'all_details': all_details,
            'uncertain': 0.4 < ensemble_score < 0.6  # Flag borderline cases
        }

        return is_accepted, details
