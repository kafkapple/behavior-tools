# E2E Test Report — behavior-tools

> Synthetic data verification of core tools.
>
> **Date**: 2026-02-08 | **Script**: `scripts/test_tools.py` | **Outputs**: `outputs/tools_test/`

---

## VideoSplitter

**Test**: Synthetic 640×480 multi-view video (30 frames, 4 colored quadrants)

| Output | Frames | Resolution |
|--------|--------|------------|
| view0_0.mp4 | 30 | 320×240 |
| view0_1.mp4 | 30 | 320×240 |
| view1_0.mp4 | 30 | 320×240 |
| view1_1.mp4 | 30 | 320×240 |

**Result**: 4 views created, correct resolution and frame count.

## FrameExtractor

- Extracted 6 frames from view 0 (interval=5, max_frames=10)
- Array extraction: shape `(5, 240, 320, 3)` uint8

**Result**: Both file and array extraction work correctly.

## ImageQualityFilter

**Test**: 10 synthetic 224×224 images with varying blur/brightness.

| Metric | Value |
|--------|-------|
| Total | 10 |
| Passed | 6 |
| Failed | 4 (blur + brightness) |

Blur detection (Laplacian variance) correctly identified intentionally blurred images.

## ImageDeduplicator

**Test**: Image #5 is a copy of image #0.

- **Method**: Perceptual hash (phash), hash_size=8, threshold=5
- **Duplicates found**: 1

**Result**: Correctly identified the duplicate pair.

---

## Backlinks

- [behavior-tools README](README.md) — Module catalog
- [behavior-lab E2E](../../behavior-lab/docs/e2e_verification.md) — Companion analysis verification

---

*behavior-tools v0.1 | E2E Test Report | 2026-02-08*
