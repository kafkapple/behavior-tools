"""CLI for SAM annotator.

TODO: Port from gpu03:~/dev/mouse-super-resolution/sam_annotator/cli.py
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="SAM Annotation Tool")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint")
    parser.add_argument("--model-type", default="vit_b", help="SAM model type")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    from .app import create_app

    app = create_app(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        share=args.share,
    )
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
