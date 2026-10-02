"""
=============================================================================
🐘 PROJECT ZOGAN — REAL-TIME DETECTION, TRACKING & RISK SYSTEM
=============================================================================

Main orchestrator. Captures video, runs YOLO detection, tracks elephants via
ByteTrack, classifies geofence zones, evaluates risk, and triggers alerts.

Uses centralized config.py and delegates rendering to ai.renderer,
detection logic to ai.detector, and model management to ai.model_manager.

Pipeline:
  CAMERA/VIDEO
       ↓
  YOLO MODEL (elephant_v1 or pretrained)
       ↓
  ELEPHANT DETECTION (conf >= threshold)
       ↓
  BYTETRACK TRACKER → Track IDs (#1, #2...)
       ↓
  POSITION HISTORY & IMAGE-SPACE MOVEMENT
       ↓
  GEOFENCING & ZONE CLASSIFICATION (simulated or live)
       ↓
  RULE-BASED RISK ENGINE (LOW / MEDIUM / HIGH / CRITICAL)
       ↓
  PERSISTENCE & COOLDOWN-GATED ALERT
       ↓
  LOCAL LOG (JSONL) + TELEGRAM (if configured)

Usage:
    python elephant_camera.py                           # Live webcam (default)
    python elephant_camera.py --source elephant.mp4     # Video file replay
    python elephant_camera.py --pretrained              # Pretrained baseline
    python elephant_camera.py --model path/to/model.pt  # Custom weights
    python elephant_camera.py --sim-lat 20.1205 --sim-lon 85.1205

Controls:
    Press 'Q' or 'q' to exit safely.

⚠️ GPS coordinates are SIMULATED unless real hardware is connected.
   Camera GPS ≠ Elephant GPS. Risk scores are rule-based prototypes.
=============================================================================
"""

import sys
import time
import argparse
import math
import random
from pathlib import Path

import cv2

# Central configuration
import config

# AI modules
from ai.tracking import (
    ElephantTracker,
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
    DIRECTION_UP,
    DIRECTION_DOWN,
)
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
from ai.renderer import (
    draw_bounding_box,
    draw_hud,
    draw_alert_banner,
)
from ai.model_manager import resolve_model_path

# Alert system
from alerts import create_alert_event, dispatch_alert

# YOLO import
from ultralytics import YOLO

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# =============================================================================
# 🔄 BACKWARD COMPATIBILITY EXPORTS & CONSTANTS
# =============================================================================
MODEL_PATH = config.PRETRAINED_MODEL_PATH
CUSTOM_MODEL_PATH = config.CUSTOM_MODEL_PATH
FALLBACK_CUSTOM_PATH = config.FALLBACK_CUSTOM_PATH
PRETRAINED_MODEL_PATH = config.PRETRAINED_MODEL_PATH
USE_CUSTOM_MODEL = config.USE_CUSTOM_MODEL
TARGET_CLASS = config.TARGET_CLASS
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
REQUIRED_DETECTIONS = config.REQUIRED_DETECTIONS
ALERT_COOLDOWN_SECONDS = config.ALERT_COOLDOWN_SECONDS
ALERT_BANNER_DURATION_SECONDS = config.ALERT_BANNER_DURATION_SECONDS
CAMERA_INDEX = config.CAMERA_INDEX
MOVEMENT_THRESHOLD_PIXELS = config.MOVEMENT_THRESHOLD_PIXELS
TRACKER_CONFIG = config.TRACKER_CONFIG
COLOR_ALERT_RED = config.COLOR_ALERT_RED
COLOR_WARN_YELLOW = config.COLOR_WARN_YELLOW
COLOR_SAFE_GREEN = config.COLOR_SAFE_GREEN
COLOR_OTHER_OBJ = config.COLOR_OTHER_OBJ
COLOR_TEXT_WHITE = config.COLOR_TEXT_WHITE
COLOR_OVERLAY_BG = config.COLOR_OVERLAY_BG

# Backward compatibility function alias
resolve_active_model = resolve_model_path


# =============================================================================
# 🚨 ALERT HANDLER
# =============================================================================


def trigger_alert(max_confidence, tracked_info=None, risk_info=None):
    """
    Triggers local alert actions and dispatches confirmed incident events.
    Prints a prominent notice to the terminal with timestamp, confidence,
    detailed tracking, and geographic risk context.
    Dispatches structured AlertEvent to local logs and optionally Telegram.
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
    print(f"   Cooldown initiated: {config.ALERT_COOLDOWN_SECONDS}s")
    print("=" * 60 + "\n")

    # Dispatch structured event locally (JSONL) and remotely (Telegram)
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
# 🗺️ SIMULATED ELEPHANT COORDINATE GENERATOR
# =============================================================================


class SimulatedElephantCoordinates:
    """
    Generates simulated elephant GPS coordinates that ACTUALLY CHANGE over time
    to make geofence zone transitions and risk assessments meaningful.

    In real deployment, these would come from triangulation, depth sensors, or
    GPS collar data. This simulation creates a gradual drift to demonstrate
    the risk engine responding to spatial changes.

    ⚠️ THIS IS A SIMULATION. Camera GPS ≠ Elephant GPS.
    """

    def __init__(self, base_lat: float, base_lon: float):
        self._base_lat = base_lat
        self._base_lon = base_lon
        self._current_lat = base_lat
        self._current_lon = base_lon
        self._step = 0
        # Random drift direction (simulates unpredictable movement)
        self._drift_angle = random.uniform(0, 2 * math.pi)
        # Meters per step (small drift per frame, ~2-5m)
        self._drift_rate = random.uniform(0.000015, 0.000035)  # degrees per step

    def update(self) -> tuple:
        """
        Returns the next simulated coordinate with gradual spatial drift.

        The elephant slowly moves in a semi-random direction, occasionally
        changing course. This creates realistic zone transitions over time.
        """
        self._step += 1

        # Occasionally shift drift direction (every ~100 steps)
        if self._step % 100 == 0:
            self._drift_angle += random.uniform(-0.5, 0.5)

        # Apply small drift
        self._current_lat += self._drift_rate * math.cos(self._drift_angle)
        self._current_lon += self._drift_rate * math.sin(self._drift_angle)

        return self._current_lat, self._current_lon

    def reset(self, lat: float, lon: float) -> None:
        """Resets to a new base position."""
        self._current_lat = lat
        self._current_lon = lon
        self._step = 0


# =============================================================================
# 🔍 MAIN DETECTION & TRACKING LOOP
# =============================================================================


def main(
    use_custom=None,
    model_override=None,
    camera_idx=None,
    conf_thresh=None,
    source=None,
    movement_threshold=None,
    tracker_config=None,
    no_show=False,
    max_frames=None,
    sim_mode=True,
    sim_lat=None,
    sim_lon=None,
    min_alert_risk=None,
):
    """
    Main detection, tracking, geofencing, and risk analysis execution loop.
    Supports live webcam feeds or recorded video file replay.
    """
    # Apply config defaults for any unspecified parameters
    if camera_idx is None:
        camera_idx = config.CAMERA_INDEX
    if conf_thresh is None:
        conf_thresh = config.CONFIDENCE_THRESHOLD
    if movement_threshold is None:
        movement_threshold = config.MOVEMENT_THRESHOLD_PIXELS
    if tracker_config is None:
        tracker_config = config.TRACKER_CONFIG

    # 1. Resolve model path
    use_custom_flag = config.USE_CUSTOM_MODEL if use_custom is None else use_custom
    model_path, model_label = resolve_model_path(use_custom_flag, model_override)

    if not model_path or not Path(model_path).exists():
        print("\n" + "=" * 60)
        print("ERROR: Model not found.")
        print(f"Expected: {config.CUSTOM_MODEL_PATH} or {config.FALLBACK_CUSTOM_PATH}")
        print("Train first (python ai/train.py) or use: python elephant_camera.py --pretrained")
        print("=" * 60 + "\n")
        return False

    # 2. Load YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"\nERROR: Failed to load YOLO model from '{model_path}': {e}")
        return False

    # 3. Verify target class exists
    model_classes = model.names
    target_class_found = False
    target_class_id = None
    for cid, cname in model_classes.items():
        if cname.lower() == config.TARGET_CLASS.lower():
            target_class_found = True
            target_class_id = cid
            break

    if not target_class_found:
        print("\n" + "=" * 60)
        print(f"ERROR: Model does not contain '{config.TARGET_CLASS}' class.")
        print(f"Model: {model_path}")
        print(f"Available: {model_classes}")
        print("=" * 60 + "\n")
        return False

    # 4. Resolve input source
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

    # Startup banner
    print("=" * 60)
    print("🐘 PROJECT ZOGAN — DETECTION, TRACKING & RISK SYSTEM")
    print(f"Source:               {source_desc}")
    print(f"Model:                {model_label}")
    print(f"Weights:              {model_path}")
    print(f"Target Class:         {config.TARGET_CLASS} (Class ID: {target_class_id})")
    print(f"Confidence Threshold: {int(conf_thresh * 100)}%")
    print(f"Tracker:              ByteTrack ({tracker_config})")
    print(f"Movement Threshold:   {movement_threshold} pixels")
    print(f"Geofencing Mode:      {'SOFTWARE SIMULATION' if sim_mode else 'DISABLED'}")
    if sim_mode:
        print(f"Sim Camera GPS:       ({config.CAMERA_LATITUDE:.6f}, {config.CAMERA_LONGITUDE:.6f})")
    print(f"Required Detections:  {config.REQUIRED_DETECTIONS} frames")
    print(f"Alert Cooldown:       {config.ALERT_COOLDOWN_SECONDS}s")
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
            print("Check that your camera is connected and not in use by another application.")
        return False

    print("[INFO] Stream initialized. Press 'Q' to exit.\n")

    # Initialize tracking
    tracker = ElephantTracker(
        max_history=config.MAX_POSITION_HISTORY,
        movement_threshold=movement_threshold,
        max_lost_frames=config.MAX_LOST_FRAMES,
        window_size=config.SMOOTHING_WINDOW_FRAMES,
    )
    fallback_track_id = 1

    # Initialize geofencing and risk engine
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

    # Simulated elephant coordinate generator (ACTUALLY MOVES each frame)
    base_sim_lat = sim_lat if sim_lat is not None else (config.CAMERA_LATITUDE - 0.0020)
    base_sim_lon = sim_lon if sim_lon is not None else (config.CAMERA_LONGITUDE - 0.0020)
    sim_coords = SimulatedElephantCoordinates(base_sim_lat, base_sim_lon)

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
                    print(f"\n[INFO] End of video file '{input_source}'. Playback finished.")
                else:
                    print("\nWARNING: Failed to read frame from camera.")
                break

            frame_number += 1
            if max_frames and frame_number > max_frames:
                print(f"\n[INFO] Reached max_frames ({max_frames}). Halting.")
                break

            current_time = time.time()

            # Calculate running FPS (exponential moving average)
            delta_time = current_time - prev_frame_time
            prev_frame_time = current_time
            if delta_time > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta_time) if fps > 0 else (1.0 / delta_time)

            # Run YOLO detection & ByteTrack tracking
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

                    # Extract tracker ID
                    track_id = None
                    if getattr(box, "id", None) is not None and box.id is not None:
                        track_id = int(box.id[0])

                    if class_name.lower() == config.TARGET_CLASS.lower():
                        if confidence >= conf_thresh:
                            candidate_elephants.append({"id": track_id, "bbox": (x1, y1, x2, y2), "conf": confidence})
                        else:
                            low_conf_elephants.append({"bbox": (x1, y1, x2, y2), "conf": confidence})
                    else:
                        other_objects.append({"name": class_name, "bbox": (x1, y1, x2, y2), "conf": confidence})

            # Ensure track ID is present for all confirmed candidates
            detections_to_track = []
            for cand in candidate_elephants:
                tid = cand["id"]
                if tid is None:
                    tid = fallback_track_id
                    fallback_track_id += 1
                detections_to_track.append({"id": tid, "bbox": cand["bbox"], "conf": cand["conf"]})

            # Update tracker
            active_tracks = tracker.update(detections_to_track, frame_idx=frame_number)

            # =================================================================
            # 🗺️ GEOFENCING & RISK ASSESSMENT
            # =================================================================
            elephant_found_in_frame = len(active_tracks) > 0
            max_elephant_confidence = max([t.confidence for t in active_tracks], default=0.0)

            if elephant_found_in_frame and sim_mode:
                # Get MOVING simulated coordinates (not static!)
                sim_elephant_lat, sim_elephant_lon = sim_coords.update()

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
            cooldown_active = time_since_last_alert < config.ALERT_COOLDOWN_SECONDS
            cooldown_remaining = (
                max(0.0, config.ALERT_COOLDOWN_SECONDS - time_since_last_alert) if cooldown_active else 0.0
            )

            if consecutive_elephant_frames >= config.REQUIRED_DETECTIONS:
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
                    alert_banner_until = current_time + config.ALERT_BANNER_DURATION_SECONDS

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
                    color=config.COLOR_ALERT_RED,
                    is_target=True,
                    sub_label=sub_label,
                    trajectory=list(track.history),
                )

            # 2. Draw low-confidence candidates
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
                    color=config.COLOR_WARN_YELLOW,
                    is_target=False,
                )

            # 3. Draw other objects
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
                    color=config.COLOR_OTHER_OBJ,
                    is_target=False,
                )

            # Determine visual status
            if consecutive_elephant_frames >= config.REQUIRED_DETECTIONS:
                if current_risk_assessment and current_risk_assessment.level in (RISK_CRITICAL, RISK_HIGH):
                    status_text = f"ELEPHANT CONFIRMED — RISK: {current_risk_assessment.level}"
                    status_color = config.COLOR_ALERT_RED
                else:
                    status_text = "ELEPHANT CONFIRMED"
                    status_color = config.COLOR_ALERT_RED
            elif consecutive_elephant_frames > 0:
                status_text = f"POSSIBLE ELEPHANT ({consecutive_elephant_frames}/{config.REQUIRED_DETECTIONS})"
                status_color = config.COLOR_WARN_YELLOW
            else:
                status_text = "MONITORING (No Elephant Detected)"
                status_color = config.COLOR_SAFE_GREEN

            # Render HUD
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

            # Render alert banner if active
            if current_time < alert_banner_until:
                r_lvl = current_risk_assessment.level if current_risk_assessment else None
                draw_alert_banner(frame, risk_level=r_lvl)

            # Display video window
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
    parser = argparse.ArgumentParser(description="Real-Time Elephant Detection, Tracking & Risk System")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: camera index (0) or video file path",
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
        default=config.CAMERA_INDEX,
        help=f"Camera index (default: {config.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--thresh-px",
        type=float,
        default=config.MOVEMENT_THRESHOLD_PIXELS,
        help=f"Movement threshold in pixels (default: {config.MOVEMENT_THRESHOLD_PIXELS})",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=config.TRACKER_CONFIG,
        help=f"Tracker configuration (default: {config.TRACKER_CONFIG})",
    )
    parser.add_argument("--no-show", action="store_true", help="Run headless without display")
    parser.add_argument("--no-sim", action="store_true", help="Disable simulated GPS coordinates")
    parser.add_argument("--sim-lat", type=float, default=None, help="Simulated elephant latitude")
    parser.add_argument("--sim-lon", type=float, default=None, help="Simulated elephant longitude")
    parser.add_argument(
        "--min-risk",
        type=str,
        default=None,
        choices=[RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL],
        help="Minimum risk level to trigger alerts",
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
