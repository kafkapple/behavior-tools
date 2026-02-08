# behavior-tools Documentation (MoC)

> Map of Content — Utility collection for behavioral neuroscience research.

---

## Project Identity

Standalone tools for dataset collection, video processing, and image curation. Designed for use alongside [behavior-lab](../../behavior-lab/) (analysis framework).

---

## Module Catalog

| Module | Purpose | Status |
|--------|---------|--------|
| **splitter** | Multi-view video splitting + frame extraction | Ported |
| **collection** | Image quality filter, deduplication, clustering, scraping | Complete |
| **collection.filters** | CLIP + DINOv2 + Ensemble semantic filtering | Complete |
| **annotator** | SAM-based interactive segmentation | Partial |
| **superres** | RealESRGAN image upscaling | Partial |

### splitter/

```python
from behavior_tools.splitter import VideoSplitter, FrameExtractor

splitter = VideoSplitter(grid=(2, 2))
views = splitter.split("recording.mp4", "output/")

extractor = FrameExtractor(interval=10)
frames = extractor.extract(views[0], "frames/")
```

### collection/

```python
from behavior_tools.collection.curator import (
    ImageQualityFilter, ImageDeduplicator, ImageClusterer
)

# Quality filtering
qf = ImageQualityFilter({"blur_threshold": 100, "min_resolution": 224})
passed = qf.filter_image(path)

# Deduplication
dedup = ImageDeduplicator({"method": "phash", "similarity_threshold": 5})
duplicates = dedup.find_duplicates(image_paths)
```

### collection.filters/ (Semantic)

```python
from behavior_tools.collection.filters import CLIPFilter, DINOv2Filter, EnsembleFilter

clip = CLIPFilter(config)   # Text-prompt based
dino = DINOv2Filter(config) # Reference-image based
ensemble = EnsembleFilter([clip, dino])  # Weighted voting
```

---

## Document Map

| Document | Content |
|----------|---------|
| **[E2E Test Report](e2e_test_report.md)** | Splitter + Curator verification results |

---

## Backlinks

- **[behavior-lab](../../behavior-lab/docs/README.md)** — Companion analysis framework
- **[behavior-lab E2E](../../behavior-lab/docs/e2e_verification.md)** — Pipeline verification

---

*behavior-tools v0.1 | Created: 2026-02-08*
