"""Gradio-based annotation interface.

TODO: Port from gpu03:~/dev/mouse-super-resolution/sam_annotator/app.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def create_app(
    checkpoint_path: str,
    model_type: str = "vit_b",
    device: str = "cpu",
    share: bool = False,
):
    """Create and launch Gradio annotation app.

    Args:
        checkpoint_path: Path to SAM checkpoint
        model_type: SAM model variant
        device: Device for inference
        share: Create public Gradio link

    Returns:
        Gradio app instance
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Install: pip install gradio")

    from .backend import AnnotationBackend
    import numpy as np

    backend = AnnotationBackend(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    backend.load_model()

    def process_image(image, evt: gr.SelectData):
        """Handle click on image to generate mask."""
        if image is None:
            return None

        backend.set_image(image)
        x, y = evt.index
        mask = backend.segment(point=(x, y))

        # Overlay mask on image
        overlay = image.copy()
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        return overlay.astype(np.uint8)

    with gr.Blocks(title="SAM Annotator") as app:
        gr.Markdown("# SAM Annotation Tool")
        gr.Markdown("Click on the image to segment objects.")

        with gr.Row():
            input_image = gr.Image(label="Input", type="numpy")
            output_image = gr.Image(label="Segmented")

        input_image.select(process_image, [input_image], [output_image])

    return app
