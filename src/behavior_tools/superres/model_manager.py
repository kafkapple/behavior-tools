"""Model manager for super-resolution backends."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class ModelManager:
    """Manage SR model weights and configuration.

    Supports RealESRGAN and BasicSR backends.

    Usage:
        manager = ModelManager(model_dir="models/")
        model = manager.get_model("realesrgan_x4")
    """

    SUPPORTED_MODELS = {
        "realesrgan_x4": {"scale": 4, "backend": "realesrgan"},
        "realesrgan_x2": {"scale": 2, "backend": "realesrgan"},
    }

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict = {}

    def get_model(self, name: str = "realesrgan_x4"):
        """Get or load a super-resolution model.

        Args:
            name: Model identifier

        Returns:
            Loaded SR model
        """
        if name in self._loaded:
            return self._loaded[name]

        if name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(self.SUPPORTED_MODELS)}")

        info = self.SUPPORTED_MODELS[name]

        if info["backend"] == "realesrgan":
            model = self._load_realesrgan(info["scale"])
        else:
            raise ValueError(f"Unknown backend: {info['backend']}")

        self._loaded[name] = model
        return model

    def _load_realesrgan(self, scale: int):
        """Load RealESRGAN model."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError:
            raise ImportError("Install: pip install realesrgan basicsr")

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale,
        )
        upsampler = RealESRGANer(
            scale=scale,
            model_path=str(self.model_dir / f"RealESRGAN_x{scale}plus.pth"),
            model=model,
            half=False,
        )
        return upsampler

    def list_models(self) -> list[str]:
        return list(self.SUPPORTED_MODELS.keys())
