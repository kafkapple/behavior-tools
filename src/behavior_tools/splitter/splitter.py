"""Video splitter for multiview synchronized recordings.

TODO: Port from gpu03:~/dev/multiview-video-splitter/src/video_splitter.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class VideoSplitter:
    """Split a multi-view video into individual camera views.

    Expects a single video file containing multiple camera views arranged
    in a grid (e.g., 2x2 for 4 cameras).

    Usage:
        splitter = VideoSplitter(grid=(2, 2))
        splitter.split("recording.mp4", "output_dir/")
    """

    def __init__(
        self,
        grid: tuple[int, int] = (2, 2),
        output_format: str = "mp4",
        codec: str = "mp4v",
    ):
        self.grid = grid
        self.output_format = output_format
        self.codec = codec

    def split(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> list[Path]:
        """Split multi-view video into individual view files.

        Args:
            video_path: Input video path
            output_dir: Output directory for split videos
            start_frame: First frame to process
            end_frame: Last frame (None = all)

        Returns:
            List of output video file paths
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

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rows, cols = self.grid
        view_w = width // cols
        view_h = height // rows

        if end_frame is None:
            end_frame = total_frames

        # Create writers
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writers = []
        output_paths = []
        for r in range(rows):
            for c in range(cols):
                out_path = output_dir / f"{video_path.stem}_view{r}_{c}.{self.output_format}"
                w = cv2.VideoWriter(str(out_path), fourcc, fps, (view_w, view_h))
                writers.append((w, r, c))
                output_paths.append(out_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            for w, r, c in writers:
                view = frame[r * view_h : (r + 1) * view_h, c * view_w : (c + 1) * view_w]
                w.write(view)

        cap.release()
        for w, _, _ in writers:
            w.release()

        return output_paths
