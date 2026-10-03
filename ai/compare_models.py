"""
=============================================================================
🐘 PROJECT ZOGAN — MODEL COMPARISON UTILITY (Phase 3.4)
=============================================================================

Compares the baseline pretrained YOLO model (yolo26n.pt) and the fine-tuned
custom elephant model (models/elephant_v1/best.pt) on the same input image.

Compares:
- Elephant detection status
- Confidence score(s)
- Inference latency (milliseconds)
- Bounding box coordinates

Usage:
    python ai/compare_models.py elephant.jpg
    python ai/compare_models.py datasets/elephant/images/test/elephant_test_0001.jpg
    python ai/compare_models.py <image_path> --save comparison.jpg --no-show
=============================================================================
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from ultralytics import YOLO

PRETRAINED_PATH = Path("models/yolo26n.pt") if Path("models/yolo26n.pt").exists() else Path("yolo26n.pt")
CUSTOM_PATHS = [
    Path("models/elephant_v1/best.pt"),
    Path("runs/detect/elephant_v1/weights/best.pt"),
]


def resolve_custom_model(explicit_path: str = None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Custom model not found at: {explicit_path}")
    for p in CUSTOM_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("No custom elephant model found! Expected models/elephant_v1/best.pt")


def run_inference(model, image_path: Path, conf_threshold: float = 0.50):
    """Runs timed inference and extracts elephant detections."""
    t0 = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    detections = []
    annotated_frame = None

    for r in results:
        annotated_frame = r.plot()
        for b in r.boxes:
            c_id = int(b.cls[0])
            conf = float(b.conf[0])
            name = r.names.get(c_id, f"class_{c_id}").lower()
            if name == "elephant" or (len(r.names) == 1 and c_id == 0):
                box = [int(v) for v in b.xyxy[0].tolist()]
                detections.append({"conf": conf, "box": box})

    return detections, latency_ms, annotated_frame


def compare_models(
    image_path_str: str,
    custom_model_str: str = None,
    pretrained_model_str: str = str(PRETRAINED_PATH),
    conf_threshold: float = 0.50,
    save_path_str: str = None,
    show_window: bool = True,
):
    image_path = Path(image_path_str)
    if not image_path.exists():
        fixture_candidate = Path("tests/fixtures") / image_path.name
        if fixture_candidate.exists():
            image_path = fixture_candidate
        else:
            print(f"❌ Error: Image file not found: {image_path}")
            return False

    pretrained_path = Path(pretrained_model_str)
    if not pretrained_path.exists():
        fallback_pre = Path("models/yolo26n.pt") if Path("models/yolo26n.pt").exists() else Path("yolo26n.pt")
        if fallback_pre.exists():
            pretrained_path = fallback_pre
        print(f"❌ Error: Pretrained baseline not found: {pretrained_path}")
        return False

    try:
        custom_path = resolve_custom_model(custom_model_str)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False

    print("=" * 60)
    print("🐘 PROJECT ZOGAN — MODEL COMPARISON")
    print("=" * 60)
    print(f"Target Image:      {image_path}")
    print(f"Original Baseline: {pretrained_path}")
    print(f"Custom Model:      {custom_path}")
    print(f"Confidence Cutoff: {conf_threshold:.2f}")
    print("-" * 60)

    # 1. Load models
    orig_model = YOLO(str(pretrained_path))
    cust_model = YOLO(str(custom_path))

    # 2. Run inference
    orig_dets, orig_time, orig_plot = run_inference(orig_model, image_path, conf_threshold)
    cust_dets, cust_time, cust_plot = run_inference(cust_model, image_path, conf_threshold)

    # 3. Print Results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("-" * 60)
    print("ORIGINAL YOLO (yolo26n.pt - COCO Pretrained):")
    if orig_dets:
        print(f"  • Elephants Detected: {len(orig_dets)}")
        for idx, d in enumerate(orig_dets, start=1):
            print(f"    - Box #{idx}: Confidence = {d['conf'] * 100:.1f}% | Coordinates = {d['box']}")
    else:
        print("  • Elephants Detected: None (0)")
    print(f"  • Inference Latency:  {orig_time:.1f} ms")

    print("\nCUSTOM ELEPHANT MODEL (elephant_v1):")
    if cust_dets:
        print(f"  • Elephants Detected: {len(cust_dets)}")
        for idx, d in enumerate(cust_dets, start=1):
            print(f"    - Box #{idx}: Confidence = {d['conf'] * 100:.1f}% | Coordinates = {d['box']}")
    else:
        print("  • Elephants Detected: None (0)")
    print(f"  • Inference Latency:  {cust_time:.1f} ms")

    print("-" * 60)
    if orig_dets and cust_dets:
        top_orig = max(d["conf"] for d in orig_dets)
        top_cust = max(d["conf"] for d in cust_dets)
        print(
            f"Summary: Baseline top confidence = {top_orig * 100:.1f}% vs Custom top confidence = {top_cust * 100:.1f}%"
        )
    elif cust_dets and not orig_dets:
        print("Summary: Custom model detected elephant where baseline missed it.")
    elif orig_dets and not cust_dets:
        print("Summary: Baseline detected elephant where custom model missed it.")
    else:
        print("Summary: Neither model detected an elephant in this image.")
    print("=" * 60)

    # Save side-by-side visualization
    if orig_plot is not None and cust_plot is not None:
        h1, w1 = orig_plot.shape[:2]
        h2, w2 = cust_plot.shape[:2]
        target_h = max(h1, h2)
        target_w = max(w1, w2)

        p1 = cv2.resize(orig_plot, (target_w, target_h))
        p2 = cv2.resize(cust_plot, (target_w, target_h))

        # Add title headers
        cv2.putText(
            p1,
            "Baseline: yolo26n.pt",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            p2,
            "Custom: elephant_v1",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        side_by_side = np.hstack([p1, p2])

        out_path = Path(save_path_str) if save_path_str else Path.cwd() / f"comparison_{image_path.name}"
        cv2.imwrite(str(out_path), side_by_side)
        print(f"\n[OK] Side-by-side comparison image saved to: {out_path}")

        if show_window:
            try:
                cv2.imshow("Model Comparison (Left: Baseline | Right: Custom)", side_by_side)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error as e:
                print(f"[INFO] GUI display unavailable: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Compare original pretrained YOLO with custom elephant model.")
    parser.add_argument("image", type=str, help="Image path to evaluate")
    parser.add_argument("--custom", type=str, default=None, help="Path to custom model weights")
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(PRETRAINED_PATH),
        help="Path to baseline pretrained model",
    )
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence cutoff threshold")
    parser.add_argument("--save", type=str, default=None, help="Save path for comparison image")
    parser.add_argument("--no-show", action="store_true", help="Do not open GUI display window")
    args = parser.parse_args()

    compare_models(
        image_path_str=args.image,
        custom_model_str=args.custom,
        pretrained_model_str=args.baseline,
        conf_threshold=args.conf,
        save_path_str=args.save,
        show_window=not args.no_show,
    )


if __name__ == "__main__":
    main()
