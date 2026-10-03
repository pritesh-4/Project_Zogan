"""
=============================================================================
🐘 PROJECT ZOGAN — IMAGE DETECTION CLI
=============================================================================

Executable CLI utility to run elephant detection on a single image file.
Wraps the core prediction pipeline with flexible arguments and clean output.

Usage:
    python scripts/detect_image.py tests/fixtures/elephant.jpg
    python scripts/detect_image.py tests/fixtures/elephant.jpg --conf 0.60
    python scripts/detect_image.py tests/fixtures/elephant.jpg --save output.jpg --no-show
=============================================================================
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.predict_custom import predict_image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Project Zogan elephant detection on an image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="tests/fixtures/elephant.jpg",
        help="Path to target image file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Explicit path to YOLO model weights (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence cutoff threshold (0.0 - 1.0)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save annotated image to this path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable GUI window preview (headless mode)",
    )

    args = parser.parse_args()
    success = predict_image(
        image_path_str=args.image,
        weights_path_str=args.weights,
        conf_threshold=args.conf,
        save_path_str=args.save,
        show_window=not args.no_show,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
