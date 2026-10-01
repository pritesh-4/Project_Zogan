"""
=============================================================================
🐘 PROJECT ZOGAN — CUSTOM MODEL PREDICTION UTILITY (Phase 3.3)
=============================================================================

Runs single-image inference using our custom fine-tuned elephant detection model:
- Loads custom weights (models/elephant_v1/best.pt or runs/detect/elephant_v1/weights/best.pt)
- Performs inference on a target image
- Prints detection coordinates and confidence scores
- Displays or saves the annotated output image
- Clearly indicates whether an elephant is detected

Usage:
    python ai/predict_custom.py elephant.jpg
    python ai/predict_custom.py datasets/elephant/images/test/elephant_test_0001.jpg
    python ai/predict_custom.py elephant.jpg --conf 0.60 --save output.jpg --no-show
=============================================================================
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
from ultralytics import YOLO


DEFAULT_WEIGHTS_CANDIDATES = [
    Path("models/elephant_v1/best.pt"),
    Path("runs/detect/elephant_v1/weights/best.pt"),
]


def resolve_model_weights(requested_weights: str = None) -> Path:
    """Finds available custom model weights or raises a clear error."""
    if requested_weights:
        p = Path(requested_weights)
        if p.exists():
            return p
        raise FileNotFoundError(f"Requested weights file not found: {requested_weights}")

    for candidate in DEFAULT_WEIGHTS_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No custom trained model weights found! Tried:\n"
        + "\n".join(f"  - {c}" for c in DEFAULT_WEIGHTS_CANDIDATES)
        + "\nPlease run training first: python ai/train.py"
    )


def predict_image(
    image_path_str: str,
    weights_path_str: str = None,
    conf_threshold: float = 0.50,
    save_path_str: str = None,
    show_window: bool = True,
) -> bool:
    """Runs inference on a single image and prints detection summary."""
    img_path = Path(image_path_str)
    if not img_path.exists():
        print(f"❌ Error: Image file not found: {img_path}")
        return False

    try:
        weights_path = resolve_model_weights(weights_path_str)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False

    print("=" * 60)
    print("🐘 PROJECT ZOGAN — CUSTOM MODEL PREDICTION")
    print("=" * 60)
    print(f"Model Weights:      {weights_path}")
    print(f"Target Image:       {img_path}")
    print(f"Confidence Cutoff:  {conf_threshold:.2f}")
    print("-" * 60)

    model = YOLO(str(weights_path))

    # Read original image with OpenCV
    orig_frame = cv2.imread(str(img_path))
    if orig_frame is None:
        print(f"❌ Error: Failed to read image file: {img_path}")
        return False
    h_px, w_px = orig_frame.shape[:2]

    # Run YOLO inference
    results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)

    elephant_detections = []
    annotated_frame = orig_frame.copy()

    for r in results:
        for b in r.boxes:
            c_id = int(b.cls[0])
            conf = float(b.conf[0])
            c_name = r.names[c_id]
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            # In single-class model, class 0 is elephant
            if c_name == "elephant" or c_id == 0:
                elephant_detections.append((conf, (x1, y1, x2, y2)))

                # Draw bounding box and label
                color = (0, 0, 255) if conf >= 0.70 else (0, 215, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

                label_text = f"ELEPHANT: {conf * 100:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), baseline = cv2.getTextSize(label_text, font, 0.6, 2)
                cv2.rectangle(
                    annotated_frame,
                    (x1, max(0, y1 - th - 8)),
                    (x1 + tw + 8, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x1 + 4, y1 - 4),
                    font,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    print(f"Image Dimensions:   {w_px}x{h_px} pixels")
    if elephant_detections:
        print(f"\n🚨 ELEPHANT DETECTED! ({len(elephant_detections)} found):")
        for idx, (conf, (x1, y1, x2, y2)) in enumerate(elephant_detections, start=1):
            print(f"  • Detection #{idx}: Confidence = {conf * 100:.1f}% | Box = [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("\nℹ️  NO ELEPHANT DETECTED (No detections above confidence cutoff).")

    # Determine save path
    if save_path_str:
        out_path = Path(save_path_str)
    else:
        out_path = Path.cwd() / f"predicted_{img_path.name}"

    cv2.imwrite(str(out_path), annotated_frame)
    print(f"\n[OK] Annotated prediction saved to: {out_path}")

    # Display window if interactive
    if show_window:
        try:
            cv2.imshow(f"Prediction: {img_path.name}", annotated_frame)
            print("Press any key in image window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[INFO] GUI display not available: {e}")

    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run single-image prediction using custom trained elephant YOLO model."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the image to run detection on",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to custom model weights (default: models/elephant_v1/best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence cutoff threshold (default: 0.50)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output image save path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open GUI display window",
    )
    args = parser.parse_args()

    success = predict_image(
        image_path_str=args.image,
        weights_path_str=args.weights,
        conf_threshold=args.conf,
        save_path_str=args.save,
        show_window=not args.no_show,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
