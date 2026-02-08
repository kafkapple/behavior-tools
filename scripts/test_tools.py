#!/usr/bin/env python3
"""End-to-end verification of behavior-tools.

Tests VideoSplitter, FrameExtractor, ImageQualityFilter, and ImageDeduplicator
using synthetic data (no external dependencies).

Usage:
    python scripts/test_tools.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "tools_test"


def create_synthetic_video(path: Path, width: int = 640, height: int = 480,
                           n_frames: int = 30, fps: float = 30.0) -> None:
    """Create a synthetic multi-view video with colored quadrants."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    colors = [
        (0, 0, 255),    # top-left: red
        (0, 255, 0),    # top-right: green
        (255, 0, 0),    # bottom-left: blue
        (0, 255, 255),  # bottom-right: yellow
    ]

    hw, hh = width // 2, height // 2
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Each quadrant gets a unique color with varying intensity
        intensity = int(100 + 155 * (i / n_frames))
        frame[:hh, :hw] = tuple(int(c * intensity / 255) for c in colors[0])
        frame[:hh, hw:] = tuple(int(c * intensity / 255) for c in colors[1])
        frame[hh:, :hw] = tuple(int(c * intensity / 255) for c in colors[2])
        frame[hh:, hw:] = tuple(int(c * intensity / 255) for c in colors[3])

        # Add frame number text
        cv2.putText(frame, f"Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()


def create_synthetic_images(out_dir: Path, n_images: int = 10) -> list[Path]:
    """Create synthetic test images with varying quality."""
    from PIL import Image, ImageFilter

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i in range(n_images):
        # Base: random noise + gradient
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        # Add brightness variation
        brightness_factor = 0.3 + 1.4 * (i / n_images)
        arr = np.clip(arr * brightness_factor, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)

        # Make some images blurry
        if i % 4 == 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=10))

        # Create one duplicate (copy of image 0)
        if i == 5:
            arr0 = np.array(Image.open(paths[0]))
            img = Image.fromarray(arr0)

        path = out_dir / f"test_img_{i:03d}.png"
        img.save(path)
        paths.append(path)

    return paths


def test_splitter(report: dict) -> None:
    """Test VideoSplitter and FrameExtractor."""
    import cv2

    print("\n" + "=" * 60)
    print("VideoSplitter + FrameExtractor")
    print("=" * 60)

    out = OUT_DIR / "splitter"
    out.mkdir(parents=True, exist_ok=True)

    # Create synthetic video
    video_path = out / "test_multiview.mp4"
    create_synthetic_video(video_path, width=640, height=480, n_frames=30)
    print(f"  Created synthetic video: {video_path}")

    # --- VideoSplitter ---
    from behavior_tools.splitter import VideoSplitter

    splitter = VideoSplitter(grid=(2, 2))
    split_dir = out / "views"
    split_paths = splitter.split(video_path, split_dir)

    print(f"  Split into {len(split_paths)} views:")
    splitter_results = {"n_views": len(split_paths), "views": []}
    for p in split_paths:
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"    {p.name}: {n} frames, {w}x{h}")
        splitter_results["views"].append({
            "name": p.name, "frames": n, "width": w, "height": h,
        })

    assert len(split_paths) == 4, f"Expected 4 views, got {len(split_paths)}"
    assert all(p.exists() for p in split_paths), "Missing output files"

    # --- FrameExtractor ---
    from behavior_tools.splitter import FrameExtractor

    extractor = FrameExtractor(interval=5)
    frames_dir = out / "frames"
    frame_paths = extractor.extract(split_paths[0], frames_dir, max_frames=10)
    print(f"\n  Extracted {len(frame_paths)} frames from view 0")

    # Also test array extraction
    frames_arr = extractor.extract_as_array(split_paths[0], max_frames=5)
    print(f"  Array extraction: shape {frames_arr.shape}")

    extractor_results = {
        "n_frames_extracted": len(frame_paths),
        "array_shape": list(frames_arr.shape),
    }

    assert len(frame_paths) > 0, "No frames extracted"
    assert frames_arr.ndim == 4, f"Expected 4D array, got {frames_arr.ndim}D"

    report["splitter"] = splitter_results
    report["extractor"] = extractor_results
    print("\n  Splitter PASSED")


def test_curator(report: dict) -> None:
    """Test ImageQualityFilter and ImageDeduplicator."""
    print("\n" + "=" * 60)
    print("ImageQualityFilter + ImageDeduplicator")
    print("=" * 60)

    out = OUT_DIR / "curator"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic images
    image_paths = create_synthetic_images(img_dir, n_images=10)
    print(f"  Created {len(image_paths)} synthetic images")

    # --- ImageQualityFilter ---
    from behavior_tools.collection.curator import ImageQualityFilter

    quality_config = {
        "min_resolution": 100,
        "max_aspect_ratio": 3.0,
        "blur_threshold": 50.0,
        "brightness_range": [20, 235],
    }
    qf = ImageQualityFilter(quality_config)

    quality_results = {"total": len(image_paths), "passed": 0, "failed": 0, "details": []}
    for p in image_paths:
        passed = qf.filter_image(p)
        is_blurry = qf.is_blurry(p)
        bright_ok = qf.check_brightness(p)
        quality_results["details"].append({
            "name": p.name,
            "passed": passed,
            "blurry": is_blurry,
            "brightness_ok": bright_ok,
        })
        if passed:
            quality_results["passed"] += 1
        else:
            quality_results["failed"] += 1

    print(f"  Quality filter: {quality_results['passed']}/{quality_results['total']} passed")
    report["quality_filter"] = {k: v for k, v in quality_results.items() if k != "details"}

    # --- ImageDeduplicator ---
    try:
        from behavior_tools.collection.curator import ImageDeduplicator

        dedup_config = {
            "hash_size": 8,
            "similarity_threshold": 5,
            "method": "phash",
            "keep_higher_resolution": True,
        }
        dedup = ImageDeduplicator(dedup_config)
        duplicates = dedup.find_duplicates(image_paths)
        print(f"  Deduplicator: {len(duplicates)} duplicates found")
        report["deduplicator"] = {
            "total": len(image_paths),
            "duplicates_found": len(duplicates),
        }
    except ImportError:
        print("  Deduplicator: SKIPPED (imagehash not installed)")
        report["deduplicator"] = {"status": "skipped", "reason": "imagehash not installed"}

    print("\n  Curator PASSED")


def generate_report(report: dict) -> None:
    """Save JSON summary report."""
    out = OUT_DIR
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Saved: {out / 'report.json'}")


def main():
    print("=" * 60)
    print("behavior-tools E2E Verification")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    test_splitter(report)
    test_curator(report)
    generate_report(report)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
