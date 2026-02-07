"""Annotation backend using SAM (Segment Anything Model).

TODO: Port from gpu03:~/dev/mouse-super-resolution/sam_annotator/annotator.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class AnnotationBackend:
    """SAM-based annotation backend for interactive segmentation.

    Usage:
        backend = AnnotationBackend(model_type="vit_b")
        backend.load_model()
        mask = backend.segment(image, point=(100, 200))
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._predictor = None

    def load_model(self) -> None:
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError("Install: pip install segment-anything")

        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path required for SAM model")

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(self.device)
        self._predictor = SamPredictor(sam)

    def set_image(self, image: np.ndarray) -> None:
        """Set image for segmentation.

        Args:
            image: (H, W, 3) RGB uint8 image
        """
        if self._predictor is None:
            self.load_model()
        self._predictor.set_image(image)

    def segment(
        self,
        point: tuple[int, int] | None = None,
        box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Generate segmentation mask.

        Args:
            point: (x, y) point prompt
            box: (x1, y1, x2, y2) box prompt

        Returns:
            (H, W) boolean mask
        """
        if self._predictor is None:
            raise RuntimeError("Call load_model() and set_image() first")

        point_coords = None
        point_labels = None
        box_input = None

        if point is not None:
            point_coords = np.array([[point[0], point[1]]])
            point_labels = np.array([1])

        if box is not None:
            box_input = np.array(box)

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_input,
            multimask_output=True,
        )

        # Return highest scoring mask
        best_idx = np.argmax(scores)
        return masks[best_idx]
