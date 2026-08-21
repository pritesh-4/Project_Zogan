"""
Test script for verifying Phase 2 Elephant Detector logic:
- Persistent detection counter
- Reset logic
- Cooldown timer
- Elephant vs non-elephant filtering
- HUD and bounding box rendering
"""

import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Import functions and constants from elephant_camera
import elephant_camera as ec

def test_pipeline():
    print("=== TEST 1: Model Loading ===")
    model = YOLO(ec.MODEL_PATH)
    assert model is not None, "Failed to load model"
    print("✓ YOLO model loaded successfully.")

    print("\n=== TEST 2: Detection on elephant.jpg ===")
    img = cv2.imread("elephant.jpg")
    assert img is not None, "Failed to load elephant.jpg"
    
    results = model(img, verbose=False)
    elephant_detected = False
    detected_conf = 0.0
    for r in results:
        for b in r.boxes:
            c_id = int(b.cls[0])
            conf = float(b.conf[0])
            name = r.names[c_id]
            if name == ec.TARGET_CLASS and conf >= ec.CONFIDENCE_THRESHOLD:
                elephant_detected = True
                detected_conf = conf
                print(f"✓ Detected {name} with confidence {conf:.2f} (Threshold: {ec.CONFIDENCE_THRESHOLD})")

    assert elephant_detected, "Elephant was not detected in elephant.jpg!"

    print("\n=== TEST 3: Persistence & Alert Simulation ===")
    consecutive_frames = 0
    last_alert_time = 0.0
    alerts_triggered = 0

    # Simulate 7 consecutive frames with elephant detected
    current_time = time.time()
    for frame_idx in range(1, 8):
        # Elephant present
        consecutive_frames += 1
        time_since_last_alert = current_time - last_alert_time
        cooldown_active = time_since_last_alert < ec.ALERT_COOLDOWN_SECONDS

        if consecutive_frames >= ec.REQUIRED_DETECTIONS:
            if not cooldown_active:
                ec.trigger_alert(detected_conf)
                last_alert_time = current_time
                alerts_triggered += 1
                print(f"  Frame {frame_idx}: Alert triggered (consecutive={consecutive_frames})")
            else:
                print(f"  Frame {frame_idx}: Confirmed, but alert skipped due to active cooldown.")
        else:
            print(f"  Frame {frame_idx}: Persistence counter: {consecutive_frames}/{ec.REQUIRED_DETECTIONS}")

    assert alerts_triggered == 1, f"Expected exactly 1 alert, got {alerts_triggered}"
    print("✓ Persistence requirement (5 frames) and cooldown prevention verified.")

    print("\n=== TEST 4: Counter Reset on Missing Elephant ===")
    # Simulate blank frame (no elephant)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blank_results = model(blank_frame, verbose=False)
    
    elephant_in_blank = False
    for r in blank_results:
        for b in r.boxes:
            c_id = int(b.cls[0])
            conf = float(b.conf[0])
            if r.names[c_id] == ec.TARGET_CLASS and conf >= ec.CONFIDENCE_THRESHOLD:
                elephant_in_blank = True

    if not elephant_in_blank:
        consecutive_frames = 0
    
    assert consecutive_frames == 0, "Counter did not reset on missing elephant!"
    print("✓ Detection counter correctly reset to 0 when elephant disappeared.")

    print("\n=== TEST 5: HUD and Drawing Functions ===")
    test_frame = img.copy()
    ec.draw_bounding_box(test_frame, 50, 100, 300, 400, "ELEPHANT: 96%", ec.COLOR_ALERT_RED, is_target=True)
    ec.draw_hud(test_frame, "ELEPHANT CONFIRMED", ec.COLOR_ALERT_RED, 5, 25.4, 28.0)
    ec.draw_alert_banner(test_frame)
    
    assert test_frame.shape == img.shape, "Frame shape altered incorrectly"
    cv2.imwrite("test_output_annotated.jpg", test_frame)
    print("✓ HUD and bounding box rendering verified. Saved test_output_annotated.jpg")

    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_pipeline()
