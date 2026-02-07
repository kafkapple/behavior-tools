"""Image upscaling using super-resolution models.

TODO: Port from gpu03:~/dev/mouse-super-resolution/src/upscale.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .model_manager import ModelManager


class Upscaler:
    """Upscale images using super-resolution models.

    Usage:
        upscaler = Upscaler(scale=4)
        upscaler.upscale_image("input.jpg", "output.jpg")
        upscaler.upscale_directory("input_dir/", "output_dir/")
    """

    def __init__(
        self,
        scale: int = 4,
        model_name: str = "realesrgan_x4",
        model_dir: str | Path = "models",
    ):
        self.scale = scale
        self.model_name = model_name
        self.manager = ModelManager(model_dir)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = self.manager.get_model(self.model_name)

    def upscale_image(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Upscale a single image.

        Args:
            input_path: Source image path
            output_path: Destination path

        Returns:
            Output path
        """
        import cv2

        self._ensure_model()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")

        output, _ = self._model.enhance(img, outscale=self.scale)
        cv2.imwrite(str(output_path), output)
        return output_path

    def upscale_array(self, image: np.ndarray) -> np.ndarray:
        """Upscale a numpy array image.

        Args:
            image: (H, W, C) BGR uint8 image

        Returns:
            Upscaled (H*scale, W*scale, C) image
        """
        self._ensure_model()
        output, _ = self._model.enhance(image, outscale=self.scale)
        return output

    def upscale_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> list[Path]:
        """Upscale all images in a directory.

        Args:
            input_dir: Source directory
            output_dir: Destination directory
            extensions: File extensions to process

        Returns:
            List of output paths
        """
        from tqdm import tqdm

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in extensions]
        results = []

        for f in tqdm(files, desc="Upscaling"):
            out = output_dir / f.name
            self.upscale_image(f, out)
            results.append(out)

        return results
