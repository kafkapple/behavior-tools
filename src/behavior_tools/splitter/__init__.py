"""Multiview video splitter: split synchronized multi-camera videos.

Ported from multiview-video-splitter (server).
SCP required: scp -r gpu03:~/dev/multiview-video-splitter/src/* .
"""
from .splitter import VideoSplitter
from .extractor import FrameExtractor

__all__ = ["VideoSplitter", "FrameExtractor"]
