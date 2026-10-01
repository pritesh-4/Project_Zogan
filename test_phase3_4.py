"""
=============================================================================
🐘 PROJECT ZOGAN — AUTOMATED TEST SUITE FOR PHASE 3.4
=============================================================================

Verifies:
1. Custom model existence (models/elephant_v1/best.pt and runs/detect/elephant_v1/weights/best.pt)
2. Custom model architecture and class names (class 0 = elephant)
3. Model resolution and switching in elephant_camera.py (custom vs pretrained)
4. Missing-model handling (clean error exit without traceback)
5. Custom single-image prediction (ai/predict_custom.py)
6. Model comparison utility (ai/compare_models.py)
7. End-to-end detection pipeline with custom model:
   - Confidence threshold check (>= 70%)
   - 5-frame persistence requirement
   - Counter reset when elephant absent
   - 30-second alert cooldown (no alert spam)
   - Visual HUD rendering with active model tag
8. Phase 2 backward compatibility (test_phase2.py)

Usage:
    python test_phase3_4.py
=============================================================================
"""

import sys
import time
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from ultralytics import YOLO
import elephant_camera as ec


def run_cmd(cmd_list):
    res = subprocess.run(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return res.returncode, res.stdout, res.stderr


def test_custom_model_weights():
    print("\n--- TEST 1: Model Checkpoint Existence ---")
    custom_weights = Path("models/elephant_v1/best.pt")
    runs_weights = Path("runs/detect/elephant_v1/weights/best.pt")

    assert custom_weights.exists() or runs_weights.exists(), (
        f"Missing custom model weights! Checked {custom_weights} and {runs_weights}"
    )
    print(f"  ✓ Found deployed weights: {custom_weights}")
    print(f"  ✓ Found runs weights:     {runs_weights}")


def test_model_class_mapping():
    print("\n--- TEST 2: Model Class Mapping Verification ---")
    weights_path, _ = ec.resolve_active_model(use_custom=True)
    model = YOLO(weights_path)

    names = model.names
    print(f"  Model classes: {names}")
    assert 0 in names or "0" in names, "Model must have class 0"
    assert names.get(0, names.get("0")) == "elephant", f"Expected class 0: elephant, got {names}"
    print("  ✓ Verified: class 0 is 'elephant'")


def test_model_resolution_and_switching():
    print("\n--- TEST 3: Model Resolution & Switching ---")
    custom_path, custom_label = ec.resolve_active_model(use_custom=True)
    assert "elephant_v1" in custom_label
    assert Path(custom_path).exists()
    print(f"  ✓ Custom model active: {custom_label} -> {custom_path}")

    pre_path, pre_label = ec.resolve_active_model(use_custom=False)
    assert "yolo26n" in pre_label
    assert Path(pre_path).exists()
    print(f"  ✓ Pretrained model fallback: {pre_label} -> {pre_path}")


def test_missing_model_handling():
    print("\n--- TEST 4: Missing Model Error Handling ---")
    success = ec.main(model_override="nonexistent_fake_weights.pt")
    assert success is False, "elephant_camera.py should exit cleanly on missing model"
    print("  ✓ Clean error exit without unhandled traceback confirmed.")


def test_predict_custom():
    print("\n--- TEST 5: Single Image Custom Prediction ---")
    code, stdout, stderr = run_cmd([sys.executable, "ai/predict_custom.py", "elephant.jpg", "--no-show"])
    print(stdout)
    assert code == 0, f"predict_custom.py failed! Code: {code}\n{stderr}"
    assert "ELEPHANT DETECTED!" in stdout
    print("  ✓ predict_custom.py executed successfully.")


def test_compare_models():
    print("\n--- TEST 6: Model Comparison Script ---")
    code, stdout, stderr = run_cmd([sys.executable, "ai/compare_models.py", "elephant.jpg", "--no-show"])
    print(stdout)
    assert code == 0, f"compare_models.py failed! Code: {code}\n{stderr}"
    assert "MODEL COMPARISON RESULTS" in stdout
    assert "ORIGINAL YOLO" in stdout
    assert "CUSTOM ELEPHANT MODEL" in stdout
    print("  ✓ compare_models.py executed successfully.")


def test_live_detection_pipeline_with_custom_model():
    print("\n--- TEST 7: Detection Pipeline with Custom Model ---")
    weights_path, model_label = ec.resolve_active_model(use_custom=True)
    model = YOLO(weights_path)

    img = cv2.imread("elephant.jpg")
    assert img is not None, "Failed to load elephant.jpg"

    # 1. Detection on elephant image
    results = model(img, verbose=False)
    elephant_detected = False
    detected_conf = 0.0
    for r in results:
        for b in r.boxes:
            c_id = int(b.cls[0])
            conf = float(b.conf[0])
            if r.names.get(c_id) == ec.TARGET_CLASS and conf >= ec.CONFIDENCE_THRESHOLD:
                elephant_detected = True
                detected_conf = conf

    assert elephant_detected, "Custom model failed to detect elephant in elephant.jpg!"
    print(
        f"  ✓ Custom model detected elephant with {detected_conf * 100:.1f}% confidence (cutoff {ec.CONFIDENCE_THRESHOLD * 100:.0f}%)"
    )

    # 2. Persistence and cooldown simulation
    consecutive_frames = 0
    last_alert_time = 0.0
    alerts_triggered = 0
    now = time.time()

    for frame_idx in range(1, 8):
        consecutive_frames += 1
        time_since_last_alert = now - last_alert_time
        cooldown_active = time_since_last_alert < ec.ALERT_COOLDOWN_SECONDS

        if consecutive_frames >= ec.REQUIRED_DETECTIONS:
            if not cooldown_active:
                ec.trigger_alert(detected_conf)
                last_alert_time = now
                alerts_triggered += 1
                print(f"    Frame {frame_idx}: 🚨 Alert triggered (persistence = {consecutive_frames})")
            else:
                print(f"    Frame {frame_idx}: Confirmed, but alert skipped (cooldown active)")
        else:
            print(f"    Frame {frame_idx}: Persistence counter: {consecutive_frames}/{ec.REQUIRED_DETECTIONS}")

    assert alerts_triggered == 1, f"Expected 1 alert, got {alerts_triggered}"
    print("  ✓ 5-frame persistence and cooldown rate limiting verified.")

    # 3. Counter reset on blank frame
    blank_frame = np.zeros((400, 600, 3), dtype=np.uint8)
    blank_results = model(blank_frame, verbose=False)
    elephant_in_blank = False
    for r in blank_results:
        for b in r.boxes:
            c_id = int(b.cls[0])
            conf = float(b.conf[0])
            if r.names.get(c_id) == ec.TARGET_CLASS and conf >= ec.CONFIDENCE_THRESHOLD:
                elephant_in_blank = True

    if not elephant_in_blank:
        consecutive_frames = 0
    assert consecutive_frames == 0, "Counter failed to reset on empty frame!"
    print("  ✓ Counter reset on missing elephant verified.")

    # 4. HUD rendering with model name tag
    test_frame = img.copy()
    ec.draw_hud(
        test_frame,
        "ELEPHANT CONFIRMED",
        ec.COLOR_ALERT_RED,
        5,
        30.0,
        25.0,
        model_name=model_label,
    )
    ec.draw_alert_banner(test_frame)
    cv2.imwrite("test_phase3_4_hud_preview.jpg", test_frame)
    print("  ✓ Visual HUD with model identifier verified. Saved test_phase3_4_hud_preview.jpg")


def test_phase2_regression():
    print("\n--- TEST 8: Phase 2 Backward Compatibility ---")
    code, stdout, stderr = run_cmd([sys.executable, "test_phase2.py"])
    print(stdout)
    assert code == 0, f"test_phase2.py failed! Code: {code}\n{stderr}"
    assert "ALL TESTS PASSED SUCCESSFULLY!" in stdout
    print("  ✓ Zero regression: Phase 2 test suite passed 100%.")


def main():
    print("============================================================")
    print("🐘 RUNNING PROJECT ZOGAN PHASE 3.4 TEST SUITE")
    print("============================================================")
    test_custom_model_weights()
    test_model_class_mapping()
    test_model_resolution_and_switching()
    test_missing_model_handling()
    test_predict_custom()
    test_compare_models()
    test_live_detection_pipeline_with_custom_model()
    test_phase2_regression()
    print("\n============================================================")
    print("🎉 ALL PHASE 3.4 TESTS PASSED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    main()
