"""Frame extractor for video files.

TODO: Port from gpu03:~/dev/multiview-video-splitter/src/frame_extractor.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class FrameExtractor:
    """Extract frames from video files at specified intervals.

    Usage:
        extractor = FrameExtractor(interval=10)
        frames = extractor.extract("video.mp4", "frames_dir/")
    """

    def __init__(self, interval: int = 1, output_format: str = "jpg"):
        self.interval = interval
        self.output_format = output_format

    def extract(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
    ) -> list[Path]:
        """Extract frames from video.

        Args:
            video_path: Input video path
            output_dir: Output directory for frames
            start_frame: First frame to extract
            max_frames: Maximum frames to extract

        Returns:
            List of extracted frame paths
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("Install opencv: pip install opencv-python")

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        extracted = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % self.interval == 0:
                out_path = output_dir / f"frame_{frame_idx:06d}.{self.output_format}"
                cv2.imwrite(str(out_path), frame)
                extracted.append(out_path)
                count += 1

                if max_frames and count >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        return extracted

    def extract_as_array(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
    ) -> np.ndarray:
        """Extract frames as numpy array.

        Returns:
            (N, H, W, C) uint8 array
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("Install opencv: pip install opencv-python")

        cap = cv2.VideoCapture(str(Path(video_path)))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        frame_idx = start_frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % self.interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if max_frames and len(frames) >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        return np.array(frames) if frames else np.empty((0, 0, 0, 3), dtype=np.uint8)
