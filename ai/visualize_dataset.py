"""
=============================================================================
🐘 PROJECT ZOGAN — DATASET LABEL VISUALIZER (Phase 3.1)
=============================================================================

Loads an image and its corresponding YOLO label (.txt) file, calculates pixel
coordinates from normalized bounding boxes, and visualizes the annotations.

Usage:
    python ai/visualize_dataset.py datasets/elephant/images/train/elephant_001.jpg
    python ai/visualize_dataset.py <image_path> --save output.jpg
    python ai/visualize_dataset.py <image_path> --no-show

Controls:
    Press any key to close the OpenCV display window (if interactive).
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure terminal handles UTF-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed. Run: pip install opencv-python")
    sys.exit(1)


# Default class mapping for single-class elephant detection
DEFAULT_CLASS_NAMES = {0: "elephant"}

# Bounding box color palette (BGR format for OpenCV)
BOX_COLOR = (0, 255, 0)         # Vibrant green for elephant box
BADGE_BG_COLOR = (0, 200, 0)    # Green label tag background
TEXT_COLOR = (0, 0, 0)          # Black text on label badge
INFO_COLOR = (255, 255, 255)    # White text for overlay info


def find_label_path(image_path: Path) -> Path:
    """
    Finds the corresponding YOLO .txt label file for a given image.
    Looks in sibling 'labels' directory first, then alongside the image.
    """
    parts = list(image_path.parts)
    
    # Check if 'images' is in path and replace with 'labels'
    if "images" in parts:
        label_parts = list(parts)
        idx = label_parts.index("images")
        label_parts[idx] = "labels"
        candidate = Path(*label_parts).with_suffix(".txt")
        if candidate.exists():
            return candidate

    # Check same directory with .txt extension
    same_dir_txt = image_path.with_suffix(".txt")
    if same_dir_txt.exists():
        return same_dir_txt

    # Return expected path in labels split directory even if it doesn't exist yet
    if "images" in parts:
        label_parts = list(parts)
        idx = label_parts.index("images")
        label_parts[idx] = "labels"
        return Path(*label_parts).with_suffix(".txt")
    
    return image_path.with_suffix(".txt")


def visualize_image(image_path_str: str, save_path_str: str = None, show_window: bool = True) -> bool:
    """
    Loads an image and its YOLO annotations, draws bounding boxes, and displays/saves the result.
    """
    image_path = Path(image_path_str)
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return False

    # Read image using OpenCV
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Error: Failed to decode image file: {image_path}")
        return False

    img_h, img_w = frame.shape[:2]
    label_path = find_label_path(image_path)

    print("=" * 60)
    print("PROJECT ZOGAN — DATASET VISUALIZER")
    print("=" * 60)
    print(f"Image:      {image_path} ({img_w}x{img_h} px)")
    print(f"Label File: {label_path}")
    print("-" * 60)

    # Check label file existence
    if not label_path.exists():
        print(f"[WARN] Notice: Label file does not exist at '{label_path}'")
        cv2.putText(
            frame,
            "NO LABEL FILE FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        # Read and parse YOLO annotations
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if len(lines) == 0:
            print("[INFO] Background Sample: Label file is empty (0 bounding boxes).")
            cv2.putText(
                frame,
                "BACKGROUND SAMPLE (0 ELEPHANTS)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            print(f"Found {len(lines)} annotation(s):")
            for idx, line in enumerate(lines, start=1):
                tokens = line.split()
                if len(tokens) != 5:
                    print(f"  [X] Line {idx}: Malformed line '{line}'")
                    continue

                try:
                    class_id = int(tokens[0])
                    xc = float(tokens[1])
                    yc = float(tokens[2])
                    w = float(tokens[3])
                    h = float(tokens[4])
                except ValueError:
                    print(f"  [X] Line {idx}: Non-numeric values in '{line}'")
                    continue

                class_name = DEFAULT_CLASS_NAMES.get(class_id, f"class_{class_id}")

                # Convert normalized coordinates to pixel coordinates
                x1 = int(round((xc - w / 2.0) * img_w))
                y1 = int(round((yc - h / 2.0) * img_h))
                x2 = int(round((xc + w / 2.0) * img_w))
                y2 = int(round((yc + h / 2.0) * img_h))

                # Clamp to image boundaries
                x1_clamped = max(0, min(img_w - 1, x1))
                y1_clamped = max(0, min(img_h - 1, y1))
                x2_clamped = max(0, min(img_w - 1, x2))
                y2_clamped = max(0, min(img_h - 1, y2))

                print(
                    f"  Box {idx}: {class_name} (class {class_id}) | "
                    f"Norm: [xc={xc:.3f}, yc={yc:.3f}, w={w:.3f}, h={h:.3f}] | "
                    f"Pixels: [{x1_clamped}, {y1_clamped}, {x2_clamped}, {y2_clamped}]"
                )

                # Draw bounding box rectangle
                cv2.rectangle(frame, (x1_clamped, y1_clamped), (x2_clamped, y2_clamped), BOX_COLOR, 3)

                # Prepare label tag text
                label_text = f"#{idx} {class_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                # Tag background rectangle
                tag_y1 = max(0, y1_clamped - th - 10)
                tag_y2 = y1_clamped
                tag_x1 = x1_clamped
                tag_x2 = min(img_w, x1_clamped + tw + 12)

                cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), BADGE_BG_COLOR, -1)
                cv2.putText(
                    frame,
                    label_text,
                    (tag_x1 + 6, tag_y2 - 6),
                    font,
                    font_scale,
                    TEXT_COLOR,
                    thickness,
                    cv2.LINE_AA,
                )

    # Save output image
    if save_path_str:
        out_path = Path(save_path_str)
    else:
        out_path = Path.cwd() / f"annotated_{image_path.name}"

    cv2.imwrite(str(out_path), frame)
    print(f"[OK] Saved visual preview to: {out_path}")

    # Display window if requested and supported
    if show_window:
        try:
            window_name = f"Project Zogan - Dataset Visualizer: {image_path.name}"
            cv2.imshow(window_name, frame)
            print("Press any key in the window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[INFO] GUI window unavailable in this environment: {e}")

    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO bounding box annotations on a dataset image."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the image file to visualize",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the annotated output image",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive GUI window (useful for headless / automated runs)",
    )
    args = parser.parse_args()

    success = visualize_image(
        args.image,
        save_path_str=args.save,
        show_window=not args.no_show,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
