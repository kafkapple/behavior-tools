"""Dataset collection tools: scraping, filtering, curation, and active learning.

Ported from rodent-dataset-collection.
"""
from .filters import CLIPFilter, DINOv2Filter, EnsembleFilter
from .scraper import create_scraper
from .curator import ImageDeduplicator, ImageClusterer
from .active_learning import generate_active_learning_interface

__all__ = [
    "CLIPFilter",
    "DINOv2Filter",
    "EnsembleFilter",
    "create_scraper",
    "ImageDeduplicator",
    "ImageClusterer",
    "generate_active_learning_interface",
]
