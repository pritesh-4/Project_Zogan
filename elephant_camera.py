"""
=============================================================================
🐘 ELEPHANT DETECTOR - Real-Time Detection, Tracking & Risk System (Phase 5)
=============================================================================

This module captures live video from a webcam or video file, runs YOLO object
detection using our validated custom fine-tuned elephant model, tracks individual
elephants across frames with ByteTrack, estimates image-space movement direction,
evaluates simulated geographic coordinates against geofenced zones (Village,
Buffer, Forest), computes rule-based early-warning risk scores, requires
multi-frame persistence, and triggers rate-limited local alerts.

Pipeline Architecture:
  CAMERA/VIDEO
       ↓
  CUSTOM YOLO MODEL (elephant_v1)
       ↓
  ELEPHANT DETECTION (conf >= 0.70)
       ↓
  BYTETRACK TRACKER (Temporary IDs: #1, #2...)
       ↓
  POSITION HISTORY & IMAGE-SPACE MOVEMENT (RIGHT/LEFT/UP/DOWN/STATIONARY)
       ↓
  GEOFENCING & ZONE CLASSIFICATION (VILLAGE / BUFFER / FOREST)
       ↓
  RULE-BASED RISK ENGINE (LOW / MEDIUM / HIGH / CRITICAL)
       ↓
  PERSISTENCE & RISK-AWARE ALERT (WITH COOLDOWN)

Usage:
    python elephant_camera.py                           # Live webcam (default)
    python elephant_camera.py --source elephant.mp4     # Video file replay
    python elephant_camera.py --source 0                # Explicit webcam index
    python elephant_camera.py --pretrained              # Pretrained baseline yolo26n.pt
    python elephant_camera.py --model path/to/model.pt  # Custom weights override
    python elephant_camera.py --sim-lat 20.1205 --sim-lon 85.1205

Controls:
    Press 'Q' or 'q' to quit the application safely.

⚠️ Phase 5 Note:
    Geographic coordinates are SIMULATED for decision-support prototyping.
    The camera is at a known coordinate; an RGB camera alone does not produce
    real-world elephant GPS. The risk score is rule-based and not an ML model.
=============================================================================
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

# Central configuration
import config

# Phase 4 tracking module
from ai.tracking import (
    ElephantTracker,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MOVEMENT_THRESHOLD,
    DEFAULT_MAX_LOST_FRAMES,
    DEFAULT_SMOOTHING_WINDOW,
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
    DIRECTION_UP,
    DIRECTION_DOWN,
)

# Phase 5 Geofencing & Risk Engine
from ai.geofence import (
    classify_zone,
    get_distance_to_protected_zone,
    create_default_zones,
)
from ai.risk_engine import (
    RiskEngine,
    MovementTrendTracker,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_HIGH,
    RISK_CRITICAL,
)

# Phase 6 Event Logging & Remote Alert System
from alerts import (
    create_alert_event,
    dispatch_alert,
)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# =============================================================================
# ⚙️ CONFIGURATION PARAMETERS
# =============================================================================

# Model Selection: Set USE_CUSTOM_MODEL = True to use our fine-tuned elephant model,
# or False to fall back to the general pretrained COCO model.
USE_CUSTOM_MODEL = True

CUSTOM_MODEL_PATH = "models/elephant_v1/best.pt"
FALLBACK_CUSTOM_PATH = "runs/detect/elephant_v1/weights/best.pt"
PRETRAINED_MODEL_PATH = "yolo26n.pt"


def resolve_active_model(use_custom: bool = USE_CUSTOM_MODEL, override_path: str = None):
    """
    Resolves active model path and human-readable label with clean fallback handling.
    """
    if override_path:
        p = Path(override_path)
        return str(p), f"Custom ({p.name})"

    if use_custom:
        if Path(CUSTOM_MODEL_PATH).exists():
            return CUSTOM_MODEL_PATH, "elephant_v1 (Custom)"
        elif Path(FALLBACK_CUSTOM_PATH).exists():
            return FALLBACK_CUSTOM_PATH, "elephant_v1 (Custom)"
        else:
            return None, "elephant_v1 (Custom)"

    return PRETRAINED_MODEL_PATH, "yolo26n (Pretrained)"


_active_path, MODEL_NAME = resolve_active_model(USE_CUSTOM_MODEL)
MODEL_PATH = _active_path or CUSTOM_MODEL_PATH

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

# Tracking Configuration (Phase 4)
TRACKER_CONFIG = "bytetrack.yaml"
MOVEMENT_THRESHOLD_PIXELS = DEFAULT_MOVEMENT_THRESHOLD  # 10.0 pixels
MAX_POSITION_HISTORY = DEFAULT_MAX_HISTORY  # 20 points
MAX_LOST_FRAMES = DEFAULT_MAX_LOST_FRAMES  # 30 frames (~1.0s)
SMOOTHING_WINDOW_FRAMES = DEFAULT_SMOOTHING_WINDOW  # 5 frames

# Colors for bounding boxes and HUD (BGR format for OpenCV)
COLOR_ALERT_RED = (0, 0, 255)  # Red for confirmed elephant / alert / high risk
COLOR_WARN_YELLOW = (0, 215, 255)  # Amber/Yellow for possible elephant / medium risk
COLOR_SAFE_GREEN = (0, 255, 0)  # Green for normal monitoring / low risk
COLOR_OTHER_OBJ = (255, 180, 0)  # Cyan/Blue for other detected objects
COLOR_TEXT_WHITE = (255, 255, 255)  # White for text readability
COLOR_OVERLAY_BG = (25, 25, 25)  # Dark gray for HUD banner background


# =============================================================================
# 🎨 UI & DRAWING HELPER FUNCTIONS
# =============================================================================


def draw_bounding_box(
    frame,
    x1,
    y1,
    x2,
    y2,
    label,
    color,
    is_target=False,
    sub_label=None,
    trajectory=None,
    hud_bar_height=85,
):
    """
    Draws a styled bounding box with an informative background label tag.
    Optionally renders movement direction sub-label, center point, and trajectory trail.
    """
    thickness = 3 if is_target else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw center point and trajectory trail if available
    if trajectory and len(trajectory) > 1:
        pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [pts],
            isClosed=False,
            color=(0, 215, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if trajectory and len(trajectory) > 0:
        cx, cy = int(trajectory[-1][0]), int(trajectory[-1][1])
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1, cv2.LINE_AA)

    # Calculate text sizes for multi-line badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55 if is_target else 0.5
    text_thickness = 2 if is_target else 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)

    sub_w, sub_h = 0, 0
    if sub_label:
        sub_scale = 0.45
        (sub_w, sub_h), _ = cv2.getTextSize(sub_label, font, sub_scale, 1)

    badge_w = max(text_w, sub_w) + 14
    line_spacing = 4
    badge_height = text_h + (sub_h + line_spacing if sub_label else 0) + 12

    # Avoid top HUD bar
    if y1 - badge_height >= hud_bar_height:
        # Place label above the box
        label_y1 = y1 - badge_height
        label_y2 = y1
    else:
        # Place label inside the box, guaranteed below HUD bar
        label_y1 = max(y1, hud_bar_height + 4)
        label_y2 = label_y1 + badge_height

    label_x1 = max(0, x1)
    label_x2 = min(frame.shape[1], label_x1 + badge_w)

    # Draw label background rectangle
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    # Add a thin white border around the badge for maximum crispness
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), 1)

    # Primary label
    t1_y = label_y1 + text_h + 5
    cv2.putText(
        frame,
        label,
        (label_x1 + 6, t1_y),
        font,
        font_scale,
        COLOR_TEXT_WHITE,
        text_thickness,
        cv2.LINE_AA,
    )

    # Movement sub-label
    if sub_label:
        t2_y = t1_y + sub_h + line_spacing
        cv2.putText(
            frame,
            sub_label,
            (label_x1 + 6, t2_y),
            font,
            0.45,
            (220, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_hud(
    frame,
    status_text,
    status_color,
    detection_count,
    fps,
    cooldown_remaining,
    model_name=None,
    tracking_summary=None,
    tracked_count=0,
    risk_assessment=None,
    geo_mode="SIMULATION",
):
    """
    Draws an informative Heads-Up Display (HUD) overlay at the top of the video frame.
    Shows current monitoring state, active model version, persistence counter, FPS,
    cooldown timer, active tracked elephant movement, and Phase 5 geographic risk context.
    """
    height, width = frame.shape[:2]
    active_model_str = model_name or MODEL_NAME

    has_tracks = (tracked_count > 0) and bool(tracking_summary)
    has_risk = risk_assessment is not None and tracked_count > 0

    if has_risk:
        bar_height = 98
    elif has_tracks:
        bar_height = 80
    else:
        bar_height = 65

    # Create top status bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_height), COLOR_OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Determine responsive font scale based on frame width
    scale_factor = min(1.0, max(0.7, width / 700.0))
    status_scale = 0.68 * scale_factor
    sub_scale = 0.48 * scale_factor

    # Row 1: Status text (Left) & Model tag (Right)
    status_str = f"STATUS: {status_text}"
    cv2.putText(
        frame,
        status_str,
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        status_scale,
        status_color,
        2,
        cv2.LINE_AA,
    )

    model_tag = f"MODEL: {active_model_str}"
    (model_w, _), _ = cv2.getTextSize(model_tag, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
    model_x = max(int(width * 0.52), width - model_w - 15)
    cv2.putText(
        frame,
        model_tag,
        (model_x, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_scale,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Row 2: Persistence / Cooldown info & FPS
    if 0 < detection_count < REQUIRED_DETECTIONS:
        sub_text = f"Persistence: {detection_count}/{REQUIRED_DETECTIONS} consecutive frames"
        sub_color = COLOR_WARN_YELLOW
    elif cooldown_remaining > 0:
        sub_text = f"Cooldown Active: {cooldown_remaining:.0f}s remaining (Monitoring continues)"
        sub_color = COLOR_WARN_YELLOW
    else:
        sub_text = f"Target: {TARGET_CLASS.upper()} | Min Conf: {int(CONFIDENCE_THRESHOLD * 100)}% | Press 'Q' to exit"
        sub_color = (200, 200, 200)

    cv2.putText(
        frame,
        sub_text,
        (15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_scale,
        sub_color,
        1,
        cv2.LINE_AA,
    )

    fps_text = f"FPS: {fps:.1f}"
    (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
    cv2.putText(
        frame,
        fps_text,
        (width - fps_w - 15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Row 3: Global Tracking HUD summary (if active tracks present)
    if has_tracks:
        track_text = f"TRACKS ({tracked_count}): {tracking_summary}"
        cv2.putText(
            frame,
            track_text,
            (15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            sub_scale,
            COLOR_WARN_YELLOW,
            1,
            cv2.LINE_AA,
        )

    # Row 4: Geofencing & Risk Assessment Summary (Phase 5)
    if has_risk:
        r = risk_assessment
        risk_color = (
            COLOR_ALERT_RED
            if r.level in (RISK_CRITICAL, RISK_HIGH)
            else (COLOR_WARN_YELLOW if r.level == RISK_MEDIUM else COLOR_SAFE_GREEN)
        )
        geo_text = (
            f"GEO [{geo_mode}]: Zone: {r.zone} | Protected Dist: {r.distance_to_protected:.0f}m | "
            f"Trend: {r.trend} | Risk: {r.level} ({r.score}/100)"
        )
        cv2.putText(
            frame,
            geo_text,
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43 * scale_factor,
            risk_color,
            1,
            cv2.LINE_AA,
        )


def draw_alert_banner(frame, risk_level=None):
    """
    Renders an emergency visual alert banner across the bottom of the screen.
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
    tag = f" — RISK: {risk_level}" if risk_level else ""
    alert_msg = f"[!] ELEPHANT DETECTED — EARLY WARNING ALERT{tag} [!]"
    (text_w, text_h), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    text_x = max(30, (width - text_w) // 2)
    cv2.putText(
        frame,
        alert_msg,
        (text_x, banner_y1 + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_TEXT_WHITE,
        2,
        cv2.LINE_AA,
    )


# =============================================================================
# 🚨 ALERT HANDLER
# =============================================================================


def trigger_alert(max_confidence, tracked_info=None, risk_info=None):
    """
    Triggers local alert actions and dispatches confirmed incident events.
    Prints a prominent notice to the terminal with timestamp, confidence,
    detailed tracking, and Phase 5 geographic risk context.
    Dispatches structured AlertEvent to local logs (logs/alerts.jsonl) and Telegram.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    conf_percent = max_confidence * 100
    print("\n" + "=" * 60)
    if risk_info and risk_info.get("risk_level") in (RISK_HIGH, RISK_CRITICAL):
        print(f"🚨 [ALERT {timestamp}] ELEPHANT EARLY WARNING! (Risk: {risk_info['risk_level']})")
    else:
        print(f"🚨 [ALERT {timestamp}] ELEPHANT CONFIRMED!")

    if risk_info:
        r_level = risk_info.get("risk_level", "N/A")
        r_score = risk_info.get("risk_score", "N/A")
        r_zone = risk_info.get("zone", "N/A")
        r_dist = risk_info.get("distance_to_protected_m", "N/A")
        r_trend = risk_info.get("trend", "N/A")
        r_group = risk_info.get("group_size", 1)
        r_mode = risk_info.get("geo_mode", "SIMULATION")

        print(f"   Risk Level:     {r_level} (Score: {r_score}/100)")
        print(f"   Zone:           {r_zone}")
        print(f"   Protected Dist: {r_dist}m" if isinstance(r_dist, (int, float)) else f"   Protected Dist: {r_dist}")
        print(f"   Approach Trend: {r_trend}")
        print(f"   Group Size:     {r_group}")
        print(f"   Geo Mode:       {r_mode}")

    if tracked_info:
        for t in tracked_info:
            tid = t.get("id", t.get("track_id", "N/A"))
            c = t.get("conf", t.get("confidence", max_confidence)) * 100
            m = t.get("movement", "UNKNOWN")
            print(f"   Track ID:       #{tid}")
            print(f"   Confidence:     {c:.1f}%")
            print(f"   Movement:       {m}")
    else:
        print(f"   Confidence:     {conf_percent:.1f}%")
    print(f"   Cooldown initiated: {ALERT_COOLDOWN_SECONDS}s")
    print("=" * 60 + "\n")

    # Phase 6: Dispatch structured event locally (JSONL) and remotely (Telegram)
    try:
        event = create_alert_event(
            confidence=max_confidence,
            tracked_info=tracked_info,
            risk_info=risk_info,
        )
        return dispatch_alert(event)
    except Exception as e:
        print(f"[ALERT DISPATCH ERROR] Failed to dispatch alert: {e}")
        return None


# =============================================================================
# 🔍 MAIN DETECTION & TRACKING LOOP
# =============================================================================


def main(
    use_custom=None,
    model_override=None,
    camera_idx=CAMERA_INDEX,
    conf_thresh=CONFIDENCE_THRESHOLD,
    source=None,
    movement_threshold=MOVEMENT_THRESHOLD_PIXELS,
    tracker_config=TRACKER_CONFIG,
    no_show=False,
    max_frames=None,
    sim_mode=True,
    sim_lat=None,
    sim_lon=None,
    min_alert_risk=None,
):
    """
    Main detection, tracking, geofencing, and risk analysis execution loop.
    Supports live webcam feeds or recorded video file replay with simulated coordinates.
    """
    # Determine model configuration
    use_custom_flag = USE_CUSTOM_MODEL if use_custom is None else use_custom
    model_path, model_label = resolve_active_model(use_custom_flag, model_override)

    # 1. Missing model check
    if not model_path or not Path(model_path).exists():
        print("\n" + "=" * 60)
        print("ERROR: Custom elephant model not found.")
        print(f"Expected: {CUSTOM_MODEL_PATH} or {FALLBACK_CUSTOM_PATH}")
        print("Please train the model first (python ai/train.py)")
        print("or run with pretrained model: python elephant_camera.py --pretrained")
        print("=" * 60 + "\n")
        return False

    # 2. Load the YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"\nERROR: Failed to load YOLO model from '{model_path}': {e}")
        return False

    # 3. Verify target class exists in loaded model
    model_classes = model.names
    target_class_found = False
    target_class_id = None
    for cid, cname in model_classes.items():
        if cname.lower() == TARGET_CLASS.lower():
            target_class_found = True
            target_class_id = cid
            break

    if not target_class_found:
        print("\n" + "=" * 60)
        print(f"ERROR: The loaded model does not contain a '{TARGET_CLASS}' class.")
        print(f"Model path: {model_path}")
        print(f"Available classes: {model_classes}")
        print("=" * 60 + "\n")
        return False

    # 4. Resolve input source (Webcam index or video file)
    input_source = source if source is not None else camera_idx
    if isinstance(input_source, str) and input_source.isdigit():
        input_source = int(input_source)

    is_video_file = isinstance(input_source, str)
    if is_video_file:
        if not Path(input_source).exists():
            print(f"\nERROR: Video source file not found: {input_source}")
            return False
        source_desc = f"Video Replay: {input_source}"
    else:
        source_desc = f"Webcam Index: {input_source}"

    # Print startup banner
    print("=" * 60)
    print("🐘 PROJECT ZOGAN — DETECTION, TRACKING & RISK SYSTEM (PHASE 5)")
    print(f"Source:               {source_desc}")
    print(f"Model:                {model_label}")
    print(f"Weights:              {model_path}")
    print(f"Target Class:         {TARGET_CLASS} (Class ID: {target_class_id})")
    print(f"Confidence Threshold: {int(conf_thresh * 100)}%")
    print(f"Tracker:              ByteTrack ({tracker_config})")
    print(f"Movement Threshold:   {movement_threshold} pixels")
    print(f"Geofencing Mode:      {'SOFTWARE SIMULATION' if sim_mode else 'DISABLED'}")
    print(f"Sim Camera GPS:       ({config.CAMERA_LATITUDE:.6f}, {config.CAMERA_LONGITUDE:.6f})")
    print(f"Required Detections:  {REQUIRED_DETECTIONS} frames")
    print(f"Alert Cooldown:       {ALERT_COOLDOWN_SECONDS}s")
    print("=" * 60)

    # 5. Open video stream
    if is_video_file:
        print(f"[INFO] Opening video file '{input_source}'...")
    else:
        print(f"[INFO] Initializing webcam (camera index {input_source})...")

    camera = cv2.VideoCapture(input_source)

    if not camera.isOpened():
        print(f"ERROR: Unable to open {source_desc}.")
        if not is_video_file:
            print("Please check that your camera is connected and not used by another application.")
        return False

    print("[INFO] Stream initialized successfully. Press 'Q' to exit.\n")

    # Initialize tracking entity
    tracker = ElephantTracker(
        max_history=MAX_POSITION_HISTORY,
        movement_threshold=movement_threshold,
        max_lost_frames=MAX_LOST_FRAMES,
        window_size=SMOOTHING_WINDOW_FRAMES,
    )
    fallback_track_id = 1

    # Initialize Geofencing zones and Risk Engine (Phase 5)
    zones = create_default_zones(
        village_center=(config.VILLAGE_CENTER_LAT, config.VILLAGE_CENTER_LON),
        village_radius=config.VILLAGE_RADIUS_METERS,
        buffer_radius=config.BUFFER_RADIUS_METERS,
        forest_radius=config.FOREST_RADIUS_METERS,
    )
    risk_engine = RiskEngine()
    trend_tracker = MovementTrendTracker(
        window_size=config.TREND_WINDOW_SIZE,
        stability_threshold_meters=config.TREND_STABILITY_THRESHOLD_METERS,
    )

    # Simulated coordinate anchor for demo elephant
    base_sim_lat = sim_lat if sim_lat is not None else (config.CAMERA_LATITUDE - 0.0020)
    base_sim_lon = sim_lon if sim_lon is not None else (config.CAMERA_LONGITUDE - 0.0020)

    # State variables
    consecutive_elephant_frames = 0
    last_alert_time = 0.0
    alert_banner_until = 0.0
    prev_frame_time = time.time()
    fps = 0.0
    frame_number = 0
    current_risk_assessment = None

    try:
        while True:
            success, frame = camera.read()
            if not success:
                if is_video_file:
                    print(f"\n[INFO] End of video file '{input_source}' reached. Playback finished.")
                else:
                    print("\nWARNING: Failed to read frame from camera.")
                break

            frame_number += 1
            if max_frames and frame_number > max_frames:
                print(f"\n[INFO] Reached requested max_frames ({max_frames}). Halting.")
                break

            current_time = time.time()

            # Calculate running FPS
            delta_time = current_time - prev_frame_time
            prev_frame_time = current_time
            if delta_time > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta_time) if fps > 0 else (1.0 / delta_time)

            # Run YOLO detection & ByteTrack tracking on current frame
            # persist=True maintains track IDs across sequential frames
            results = model.track(frame, persist=True, tracker=tracker_config, verbose=False)

            candidate_elephants = []
            low_conf_elephants = []
            other_objects = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names.get(class_id, f"class_{class_id}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Extract tracker ID if assigned by ByteTrack
                    track_id = None
                    if getattr(box, "id", None) is not None and box.id is not None:
                        track_id = int(box.id[0])

                    if class_name.lower() == TARGET_CLASS.lower():
                        if confidence >= conf_thresh:
                            candidate_elephants.append(
                                {
                                    "id": track_id,
                                    "bbox": (x1, y1, x2, y2),
                                    "conf": confidence,
                                }
                            )
                        else:
                            low_conf_elephants.append(
                                {
                                    "bbox": (x1, y1, x2, y2),
                                    "conf": confidence,
                                }
                            )
                    else:
                        other_objects.append(
                            {
                                "name": class_name,
                                "bbox": (x1, y1, x2, y2),
                                "conf": confidence,
                            }
                        )

            # Ensure track ID is present for all confirmed candidates
            detections_to_track = []
            for cand in candidate_elephants:
                tid = cand["id"]
                if tid is None:
                    tid = fallback_track_id
                    fallback_track_id += 1
                detections_to_track.append(
                    {
                        "id": tid,
                        "bbox": cand["bbox"],
                        "conf": cand["conf"],
                    }
                )

            # Update high-level ElephantTracker with current frame's observations
            active_tracks = tracker.update(detections_to_track, frame_idx=frame_number)

            # =================================================================
            # 🗺️ GEOFENCING & RISK ASSESSMENT (PHASE 5)
            # =================================================================
            elephant_found_in_frame = len(active_tracks) > 0
            max_elephant_confidence = max([t.confidence for t in active_tracks], default=0.0)

            if elephant_found_in_frame and sim_mode:
                # Simulated elephant coordinates (step closer if moving, or static in buffer)
                # In simulation mode, evaluate proximity to protected village zone
                sim_elephant_lat = base_sim_lat
                sim_elephant_lon = base_sim_lon

                current_zone = classify_zone(sim_elephant_lat, sim_elephant_lon, zones)
                dist_to_protected = get_distance_to_protected_zone(sim_elephant_lat, sim_elephant_lon, zones)
                approach_trend = trend_tracker.add_observation(dist_to_protected)

                current_risk_assessment = risk_engine.evaluate(
                    zone=current_zone,
                    distance_to_protected=dist_to_protected,
                    trend=approach_trend,
                    group_size=len(active_tracks),
                    confidence=max_elephant_confidence,
                )
            elif not elephant_found_in_frame:
                current_risk_assessment = None

            # =================================================================
            # 🔄 PERSISTENT DETECTION & RESET LOGIC
            # =================================================================
            if elephant_found_in_frame:
                consecutive_elephant_frames += 1
            else:
                consecutive_elephant_frames = 0

            # =================================================================
            # 🚨 ALERT & COOLDOWN LOGIC
            # =================================================================
            time_since_last_alert = current_time - last_alert_time
            cooldown_active = time_since_last_alert < ALERT_COOLDOWN_SECONDS
            cooldown_remaining = max(0.0, ALERT_COOLDOWN_SECONDS - time_since_last_alert) if cooldown_active else 0.0

            if consecutive_elephant_frames >= REQUIRED_DETECTIONS:
                # Check optional minimum risk filter (if configured)
                should_alert = not cooldown_active
                if min_alert_risk and current_risk_assessment:
                    should_alert = should_alert and (
                        current_risk_assessment.level in (min_alert_risk, RISK_CRITICAL, RISK_HIGH)
                    )

                if should_alert:
                    tracked_alert_info = [
                        {"id": t.track_id, "conf": t.confidence, "movement": t.movement} for t in active_tracks
                    ]
                    risk_info_dict = current_risk_assessment.to_dict() if current_risk_assessment else None
                    trigger_alert(
                        max_elephant_confidence,
                        tracked_info=tracked_alert_info,
                        risk_info=risk_info_dict,
                    )
                    last_alert_time = current_time
                    alert_banner_until = current_time + ALERT_BANNER_DURATION_SECONDS

            # =================================================================
            # 🎨 RENDER VISUAL OUTPUT
            # =================================================================

            # 1. Draw confirmed tracked elephants
            for track in active_tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                label = f"ELEPHANT #{track.track_id} | {int(track.confidence * 100)}%"
                mov = track.movement
                sub_label = (
                    f"MOVING {mov}" if mov in (DIRECTION_RIGHT, DIRECTION_LEFT, DIRECTION_UP, DIRECTION_DOWN) else mov
                )

                draw_bounding_box(
                    frame=frame,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=label,
                    color=COLOR_ALERT_RED,
                    is_target=True,
                    sub_label=sub_label,
                    trajectory=list(track.history),
                )

            # 2. Draw low-confidence candidate elephants
            for low in low_conf_elephants:
                x1, y1, x2, y2 = low["bbox"]
                label = f"elephant (low conf): {int(low['conf'] * 100)}%"
                draw_bounding_box(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    label=label,
                    color=COLOR_WARN_YELLOW,
                    is_target=False,
                )

            # 3. Draw other detected objects
            for obj in other_objects:
                x1, y1, x2, y2 = obj["bbox"]
                label = f"{obj['name']}: {int(obj['conf'] * 100)}%"
                draw_bounding_box(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    label=label,
                    color=COLOR_OTHER_OBJ,
                    is_target=False,
                )

            # Determine visual status text and badge color
            if consecutive_elephant_frames >= REQUIRED_DETECTIONS:
                if current_risk_assessment and current_risk_assessment.level in (
                    RISK_CRITICAL,
                    RISK_HIGH,
                ):
                    status_text = f"ELEPHANT CONFIRMED — RISK: {current_risk_assessment.level}"
                    status_color = COLOR_ALERT_RED
                else:
                    status_text = "ELEPHANT CONFIRMED"
                    status_color = COLOR_ALERT_RED
            elif consecutive_elephant_frames > 0:
                status_text = f"POSSIBLE ELEPHANT ({consecutive_elephant_frames}/{REQUIRED_DETECTIONS})"
                status_color = COLOR_WARN_YELLOW
            else:
                status_text = "MONITORING (No Elephant Detected)"
                status_color = COLOR_SAFE_GREEN

            # Render top HUD with tracking and risk status
            draw_hud(
                frame=frame,
                status_text=status_text,
                status_color=status_color,
                detection_count=consecutive_elephant_frames,
                fps=fps,
                cooldown_remaining=cooldown_remaining,
                model_name=model_label,
                tracking_summary=tracker.get_hud_summary(),
                tracked_count=len(active_tracks),
                risk_assessment=current_risk_assessment,
                geo_mode="SIMULATION" if sim_mode else "OFF",
            )

            # Render emergency alert banner if active
            if current_time < alert_banner_until:
                r_lvl = current_risk_assessment.level if current_risk_assessment else None
                draw_alert_banner(frame, risk_level=r_lvl)

            # Display video window if not in headless mode
            if not no_show:
                cv2.imshow("Elephant Detection & Tracking - Early Warning System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q")]:
                    print("\n[INFO] User requested shutdown. Exiting...")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"\nERROR: Unexpected error during execution: {e}")
        return False
    finally:
        print("[INFO] Releasing video source and closing windows...")
        camera.release()
        if not no_show:
            cv2.destroyAllWindows()
        print("[INFO] Shutdown complete. Goodbye!")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Elephant Detection, Tracking & Risk System (Phase 5)")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: camera index (0) or video file path (video.mp4)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Force using pretrained yolo26n.pt model",
    )
    parser.add_argument("--model", type=str, default=None, help="Explicit custom weights path")
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"Camera index (default: {CAMERA_INDEX})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--thresh-px",
        type=float,
        default=MOVEMENT_THRESHOLD_PIXELS,
        help=f"Movement threshold in pixels (default: {MOVEMENT_THRESHOLD_PIXELS})",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=TRACKER_CONFIG,
        help=f"Tracker configuration (default: {TRACKER_CONFIG})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run headless without opening OpenCV window",
    )
    parser.add_argument("--no-sim", action="store_true", help="Disable simulated geographic coordinates")
    parser.add_argument("--sim-lat", type=float, default=None, help="Simulated elephant latitude")
    parser.add_argument("--sim-lon", type=float, default=None, help="Simulated elephant longitude")
    parser.add_argument(
        "--min-risk",
        type=str,
        default=None,
        choices=[RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL],
        help="Minimum risk level required to trigger alerts",
    )
    args = parser.parse_args()

    use_custom = False if args.pretrained else None
    main(
        use_custom=use_custom,
        model_override=args.model,
        camera_idx=args.camera,
        conf_thresh=args.conf,
        source=args.source,
        movement_threshold=args.thresh_px,
        tracker_config=args.tracker,
        no_show=args.no_show,
        sim_mode=not args.no_sim,
        sim_lat=args.sim_lat,
        sim_lon=args.sim_lon,
        min_alert_risk=args.min_risk,
    )
