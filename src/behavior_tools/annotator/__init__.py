"""SAM-based image annotation tool with Gradio interface.

Ported from mouse-super-resolution/sam_annotator (server).
SCP required: scp -r gpu03:~/dev/mouse-super-resolution/sam_annotator/* .
"""
from .backend import AnnotationBackend
from .app import create_app

__all__ = ["AnnotationBackend", "create_app"]
