"""Super-resolution module for mouse behavior images.

Ported from mouse-super-resolution (server).
SCP required: scp -r gpu03:~/dev/mouse-super-resolution/src/* .
"""
from .upscale import Upscaler
from .model_manager import ModelManager

__all__ = ["Upscaler", "ModelManager"]
