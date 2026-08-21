"""
=============================================================================
🐘 ELEPHANT DETECTOR - Real-Time Camera Detection System (Phase 2)
=============================================================================

This module captures live video from a webcam, runs YOLO object detection
on each frame, filters for elephants exceeding a confidence threshold,
requires persistent detections across multiple frames to eliminate false alarms,
and triggers rate-limited local alerts with an informative visual HUD.

Usage:
    python elephant_camera.py

Controls:
    Press 'Q' or 'q' to quit the application safely.
=============================================================================
"""

import time
import cv2
from ultralytics import YOLO

# =============================================================================
# ⚙️ CONFIGURATION PARAMETERS
# =============================================================================

# Path to the pretrained YOLO model weights
MODEL_PATH = "yolo26n.pt"

# Minimum confidence required to accept an elephant detection (70%)
CONFIDENCE_THRESHOLD = 0.70

# Number of consecutive frames an elephant must be detected
# before confirming and triggering an alert (prevents single-frame false alarms)
REQUIRED_DETECTIONS = 5

# Time in seconds to wait before allowing another alert (prevents alert spam)
ALERT_COOLDOWN_SECONDS = 30

# Duration in seconds to display the high-priority visual alert banner on screen
ALERT_BANNER_DURATION_SECONDS = 4.0

# Target class name to monitor
TARGET_CLASS = "elephant"

# Default webcam index (0 is usually the built-in or primary USB camera)
CAMERA_INDEX = 0

# Colors for bounding boxes and HUD (BGR format for OpenCV)
COLOR_ALERT_RED = (0, 0, 255)       # Red for confirmed elephant / alert
COLOR_WARN_YELLOW = (0, 215, 255)   # Amber/Yellow for possible elephant
COLOR_SAFE_GREEN = (0, 255, 0)      # Green for normal monitoring
COLOR_OTHER_OBJ = (255, 180, 0)     # Cyan/Blue for other detected objects
COLOR_TEXT_WHITE = (255, 255, 255)  # White for text readability
COLOR_OVERLAY_BG = (25, 25, 25)     # Dark gray for HUD banner background


# =============================================================================
# 🎨 UI & DRAWING HELPER FUNCTIONS
# =============================================================================

def draw_bounding_box(frame, x1, y1, x2, y2, label, color, is_target=False):
    """
    Draws a styled bounding box with a background label tag for high readability.
    """
    thickness = 3 if is_target else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Calculate text size for the background label badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 if is_target else 0.5
    text_thickness = 2 if is_target else 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    # Position label above bounding box (or inside if box is at the very top of frame)
    badge_height = text_h + 10
    if y1 - badge_height >= 70:  # Stay below top HUD bar
        label_y1 = y1 - badge_height
        label_y2 = y1
    else:
        label_y1 = y1
        label_y2 = y1 + badge_height

    label_x1 = x1
    label_x2 = min(frame.shape[1], x1 + text_w + 12)

    # Draw label background rectangle and label text
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    text_y = label_y2 - 6
    cv2.putText(
        frame,
        label,
        (label_x1 + 6, text_y),
        font,
        font_scale,
        COLOR_TEXT_WHITE,
        text_thickness,
        cv2.LINE_AA,
    )


def draw_hud(frame, status_text, status_color, detection_count, fps, cooldown_remaining):
    """
    Draws an informative Heads-Up Display (HUD) overlay at the top of the video frame.
    Shows current monitoring state, persistence counter, FPS, and cooldown timer.
    """
    height, width = frame.shape[:2]

    # Create top status bar background
    bar_height = 65
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_height), COLOR_OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Draw status badge / text
    cv2.putText(
        frame,
        f"STATUS: {status_text}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # Draw persistence progress info
    if 0 < detection_count < REQUIRED_DETECTIONS:
        sub_text = f"Persistence: {detection_count}/{REQUIRED_DETECTIONS} consecutive frames"
        cv2.putText(
            frame,
            sub_text,
            (15, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WARN_YELLOW,
            1,
            cv2.LINE_AA,
        )
    elif cooldown_remaining > 0:
        sub_text = f"Cooldown Active: {cooldown_remaining:.0f}s remaining (Monitoring continues)"
        cv2.putText(
            frame,
            sub_text,
            (15, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WARN_YELLOW,
            1,
            cv2.LINE_AA,
        )
    else:
        sub_text = f"Target: {TARGET_CLASS.upper()} | Min Conf: {int(CONFIDENCE_THRESHOLD * 100)}% | Press 'Q' to exit"
        cv2.putText(
            frame,
            sub_text,
            (15, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Draw FPS in top-right corner
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (width - 130, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_alert_banner(frame):
    """
    Renders an emergency visual alert banner across the bottom/center of the screen.
    """
    height, width = frame.shape[:2]
    banner_y1 = height - 70
    banner_y2 = height - 15

    # Red translucent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, banner_y1), (width - 20, banner_y2), COLOR_ALERT_RED, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Alert border
    cv2.rectangle(frame, (20, banner_y1), (width - 20, banner_y2), COLOR_TEXT_WHITE, 2)

    # Alert message text
    alert_msg = "[!] ELEPHANT DETECTED - EARLY WARNING ALERT [!]"
    (text_w, text_h), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = max(30, (width - text_w) // 2)
    cv2.putText(
        frame,
        alert_msg,
        (text_x, banner_y1 + 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT_WHITE,
        2,
        cv2.LINE_AA,
    )


# =============================================================================
# 🚨 ALERT HANDLER
# =============================================================================

def trigger_alert(max_confidence):
    """
    Triggers local alert actions when an elephant detection is confirmed.
    Prints a prominent notice to the terminal with timestamp and confidence.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    conf_percent = max_confidence * 100
    print("\n" + "=" * 60)
    print(f"🚨 [ALERT {timestamp}] ELEPHANT CONFIRMED!")
    print(f"   Confidence: {conf_percent:.1f}%")
    print(f"   Cooldown initiated: {ALERT_COOLDOWN_SECONDS}s")
    print("=" * 60 + "\n")


# =============================================================================
# 🔍 MAIN DETECTION LOOP
# =============================================================================

def main():
    print("=" * 60)
    print("🐘 Starting Elephant Detection & Early Warning System (Phase 2)")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Confidence Threshold: {int(CONFIDENCE_THRESHOLD * 100)}%")
    print(f"   Required Consecutive Detections: {REQUIRED_DETECTIONS}")
    print(f"   Alert Cooldown: {ALERT_COOLDOWN_SECONDS} seconds")
    print("=" * 60)

    # 1. Load the YOLO model
    try:
        print(f"[INFO] Loading YOLO model from '{MODEL_PATH}'...")
        model = YOLO(MODEL_PATH)
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        return

    # 2. Open the webcam
    print(f"[INFO] Initializing webcam (camera index {CAMERA_INDEX})...")
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Unable to access webcam.")
        print("Please check that your camera is connected and not used by another application.")
        return

    print("[INFO] Camera initialized successfully. Press 'Q' to exit.\n")

    # State variables for persistence, cooldown, and FPS
    consecutive_elephant_frames = 0
    last_alert_time = 0.0
    alert_banner_until = 0.0
    prev_frame_time = time.time()
    fps = 0.0

    try:
        while True:
            # Capture frame from webcam
            success, frame = camera.read()
            if not success:
                print("WARNING: Failed to read frame from camera.")
                break

            current_time = time.time()

            # Calculate running FPS
            delta_time = current_time - prev_frame_time
            prev_frame_time = current_time
            if delta_time > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta_time) if fps > 0 else (1.0 / delta_time)

            # Run YOLO detection on current frame (verbose=False keeps terminal output clean)
            results = model(frame, verbose=False)

            # Flags to track detections in the current frame
            elephant_found_in_frame = False
            max_elephant_confidence = 0.0

            # Process all detected bounding boxes
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if class_name == TARGET_CLASS:
                        if confidence >= CONFIDENCE_THRESHOLD:
                            # Valid elephant detection meeting confidence threshold
                            elephant_found_in_frame = True
                            max_elephant_confidence = max(max_elephant_confidence, confidence)

                            label = f"ELEPHANT: {int(confidence * 100)}%"
                            draw_bounding_box(frame, x1, y1, x2, y2, label, COLOR_ALERT_RED, is_target=True)
                        else:
                            # Elephant detected below threshold (display for visibility, but do not count)
                            label = f"elephant (low conf): {int(confidence * 100)}%"
                            draw_bounding_box(frame, x1, y1, x2, y2, label, COLOR_WARN_YELLOW, is_target=False)
                    else:
                        # Other detected objects (person, car, dog, etc.) - display without alerting
                        label = f"{class_name}: {int(confidence * 100)}%"
                        draw_bounding_box(frame, x1, y1, x2, y2, label, COLOR_OTHER_OBJ, is_target=False)

            # =================================================================
            # 🔄 PERSISTENT DETECTION & RESET LOGIC
            # =================================================================
            if elephant_found_in_frame:
                consecutive_elephant_frames += 1
            else:
                # Reset detection counter if elephant is no longer detected in this frame
                consecutive_elephant_frames = 0

            # =================================================================
            # 🚨 ALERT & COOLDOWN LOGIC
            # =================================================================
            time_since_last_alert = current_time - last_alert_time
            cooldown_active = time_since_last_alert < ALERT_COOLDOWN_SECONDS
            cooldown_remaining = max(0.0, ALERT_COOLDOWN_SECONDS - time_since_last_alert) if cooldown_active else 0.0

            # Check if persistence requirement is met
            if consecutive_elephant_frames >= REQUIRED_DETECTIONS:
                if not cooldown_active:
                    # Trigger a new alert
                    trigger_alert(max_elephant_confidence)
                    last_alert_time = current_time
                    alert_banner_until = current_time + ALERT_BANNER_DURATION_SECONDS
                else:
                    # Already alerted recently, maintain visual indicator without spamming terminal
                    pass

            # =================================================================
            # 🖥️ DETERMINE VISUAL HUD STATUS
            # =================================================================
            if consecutive_elephant_frames >= REQUIRED_DETECTIONS:
                status_text = "ELEPHANT CONFIRMED"
                status_color = COLOR_ALERT_RED
            elif consecutive_elephant_frames > 0:
                status_text = f"POSSIBLE ELEPHANT ({consecutive_elephant_frames}/{REQUIRED_DETECTIONS})"
                status_color = COLOR_WARN_YELLOW
            else:
                status_text = "MONITORING (No Elephant Detected)"
                status_color = COLOR_SAFE_GREEN

            # Draw HUD overlay
            draw_hud(
                frame=frame,
                status_text=status_text,
                status_color=status_color,
                detection_count=consecutive_elephant_frames,
                fps=fps,
                cooldown_remaining=cooldown_remaining,
            )

            # Draw emergency alert banner if within banner display duration
            if current_time < alert_banner_until:
                draw_alert_banner(frame)

            # Display the video frame
            cv2.imshow("Elephant Detection - Early Warning System", frame)

            # Check for quit key ('q' or 'Q')
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q")]:
                print("\n[INFO] User requested shutdown. Exiting...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"\nERROR: Unexpected error during execution: {e}")
    finally:
        # Clean up resources safely
        print("[INFO] Releasing camera and closing windows...")
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete. Goodbye!")


if __name__ == "__main__":
    main()