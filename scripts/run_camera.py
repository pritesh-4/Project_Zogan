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
    python scripts/run_camera.py                           # Live webcam (default)
    python scripts/run_camera.py --source elephant.mp4     # Video file replay
    python scripts/run_camera.py --pretrained              # Pretrained baseline
    python scripts/run_camera.py --model path/to/model.pt  # Custom weights
    python scripts/run_camera.py --sim-lat 20.1205 --sim-lon 85.1205

Controls:
    Press 'Q' or 'q' to exit safely.

⚠️ GPS coordinates are SIMULATED unless real hardware is connected.
   Camera GPS ≠ Elephant GPS. Risk scores are rule-based prototypes.
=============================================================================
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    AlertPolicy,
    EventLifecycleManager,
    MovementTrendTracker,
    RiskEngine,
)
from ai.renderer import (
    draw_bounding_box,
    draw_hud,
    draw_alert_banner,
)
from ai.model_manager import resolve_model_path

# Alert system
from alerts import (
    create_alert_event,
    dispatch_alert,
    drain_delivery_queue_async,
    get_alert_queue,
)

# Observability & System Health (Phase 8 & Phase 9)
from monitoring import (
    SUBSYSTEM_DEGRADED,
    SUBSYSTEM_FAILED,
    SUBSYSTEM_READY,
    SystemHealthMonitor,
)

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


def trigger_alert(
    max_confidence,
    tracked_info=None,
    risk_info=None,
    health_monitor=None,
    async_delivery=False,
):
    """
    Triggers local alert actions and dispatches confirmed incident events.
    Prints a prominent notice to the terminal with timestamp, confidence,
    detailed tracking, and geographic risk context.
    Dispatches structured AlertEvent to local logs and optionally Telegram.
    Local persistence strictly precedes remote notification.
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
        if risk_info.get("reasons"):
            print(f"   Primary Reason: {risk_info['reasons'][0]}")

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

    # Dispatch structured event locally (JSONL) first, then remote (Telegram)
    try:
        event = create_alert_event(
            confidence=max_confidence,
            tracked_info=tracked_info,
            risk_info=risk_info,
        )

        def _on_delivery_done(result):
            if health_monitor and result:
                health_monitor.record_delivery_result(
                    success=result.get("telegram_sent", False),
                    error=result.get("telegram_error"),
                    pending_count=result.get("pending_count", get_alert_queue().pending_count),
                )

        res = dispatch_alert(
            event,
            async_delivery=async_delivery,
            callback=_on_delivery_done if async_delivery else None,
        )

        if not async_delivery and health_monitor and res:
            health_monitor.record_delivery_result(
                success=res.get("telegram_sent", False),
                error=res.get("telegram_error"),
                pending_count=res.get("pending_count", get_alert_queue().pending_count),
            )

        return res
    except Exception as e:
        print(f"[ALERT DISPATCH ERROR] Failed to dispatch alert: {e}")
        if health_monitor:
            health_monitor.record_error("dispatcher", e)
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

    # Initialize Phase 8 System Health & Observability Monitor
    health_monitor = SystemHealthMonitor(
        fps_minimum=config.HEALTH_FPS_MINIMUM_HEALTHY,
        fps_window_size=config.HEALTH_FPS_WINDOW_SIZE,
        freshness_degraded_seconds=config.HEALTH_FRESHNESS_DEGRADED_SECONDS,
        freshness_offline_seconds=config.HEALTH_FRESHNESS_OFFLINE_SECONDS,
        max_consecutive_frame_failures=config.HEALTH_MAX_CONSECUTIVE_FRAME_FAILURES,
        max_consecutive_frame_failures_offline=config.HEALTH_MAX_CONSECUTIVE_FRAME_FAILURES_OFFLINE,
        max_consecutive_errors_degraded=config.HEALTH_MAX_CONSECUTIVE_ERRORS_DEGRADED,
        max_consecutive_errors_offline=config.HEALTH_MAX_CONSECUTIVE_ERRORS_OFFLINE,
    )

    # 1. Resolve model path
    use_custom_flag = config.USE_CUSTOM_MODEL if use_custom is None else use_custom
    model_path, model_label = resolve_model_path(use_custom_flag, model_override)

    if not model_path or not Path(model_path).exists():
        health_monitor.record_detector_status(SUBSYSTEM_FAILED, "Model file not found")
        print("\n" + "=" * 60)
        print("ERROR: Model not found.")
        print(f"Expected: {config.CUSTOM_MODEL_PATH} or {config.FALLBACK_CUSTOM_PATH}")
        print("Train first (python ai/train.py) or use: python scripts/run_camera.py --pretrained")
        print("=" * 60 + "\n")
        return False

    # 2. Load YOLO model
    try:
        model = YOLO(model_path)
        health_monitor.record_detector_status(SUBSYSTEM_READY)
    except Exception as e:
        health_monitor.record_detector_status(SUBSYSTEM_FAILED, str(e))
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
    camera_opened = camera.isOpened()
    health_monitor.record_camera_open(camera_opened, source_desc)

    if not camera_opened:
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
    health_monitor.record_tracker_status(SUBSYSTEM_READY)
    fallback_track_id = 1

    # Initialize geofencing and risk engine 2.0
    zones = create_default_zones(
        village_center=(config.VILLAGE_CENTER_LAT, config.VILLAGE_CENTER_LON),
        village_radius=config.VILLAGE_RADIUS_METERS,
        buffer_radius=config.BUFFER_RADIUS_METERS,
        forest_radius=config.FOREST_RADIUS_METERS,
    )
    risk_engine = RiskEngine(
        low_max=config.RISK_LEVEL_LOW_MAX,
        medium_max=config.RISK_LEVEL_MEDIUM_MAX,
        high_max=config.RISK_LEVEL_HIGH_MAX,
        alert_on_levels=config.ALERT_ON_LEVELS,
        confirmation_frames=config.PERSISTENCE_CONFIRMATION_FRAMES,
        critical_min_confidence=config.CRITICAL_MIN_CONFIDENCE,
        critical_max_distance=config.CRITICAL_MAX_DISTANCE_METERS,
        single_frame_risk_cap=config.SINGLE_FRAME_RISK_CAP,
        critical_requires_persistence=config.CRITICAL_REQUIRES_PERSISTENCE,
        herd_enabled=config.HERD_ENABLED,
        herd_threshold=config.HERD_THRESHOLD,
        movement_enabled=config.MOVEMENT_RISK_ENABLED,
        max_zone_points=config.MAX_ZONE_POINTS,
        max_proximity_points=config.MAX_PROXIMITY_POINTS,
        max_trend_points=config.MAX_TREND_POINTS,
        max_persistence_points=config.MAX_PERSISTENCE_POINTS,
        max_group_points=config.MAX_GROUP_POINTS,
        max_confidence_points=config.MAX_CONFIDENCE_POINTS,
        max_duration_points=config.MAX_DURATION_POINTS,
        max_history_points=config.MAX_HISTORY_POINTS,
    )
    health_monitor.record_risk_engine_status(SUBSYSTEM_READY)
    event_manager = EventLifecycleManager(
        confirmation_frames=config.PERSISTENCE_CONFIRMATION_FRAMES,
        resolution_frames=config.EVENT_RESOLUTION_FRAMES,
        resolution_seconds=config.EVENT_RESOLUTION_SECONDS,
    )
    alert_policy = AlertPolicy(
        alert_on_levels=config.ALERT_ON_LEVELS,
        cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
        allow_escalation_bypass=config.ESCALATION_COOLDOWN_BYPASS,
        confirmation_frames=config.PERSISTENCE_CONFIRMATION_FRAMES,
    )
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
    detector_consecutive_failures = 0
    detector_total_failures = 0
    last_alert_time = 0.0
    alert_banner_until = 0.0
    prev_frame_time = time.time()
    fps = 0.0
    frame_number = 0
    current_risk_assessment = None

    try:
        while True:
            current_time = time.time()
            success, frame = camera.read()
            health_monitor.record_frame_read(success, now=current_time)
            health_monitor.record_heartbeat(now=current_time)

            if not success:
                if is_video_file:
                    snap = health_monitor.evaluate(current_time)
                    transition = health_monitor.check_transition(snap)
                    if transition:
                        old_st, new_st, reason = transition
                        print(f"\n[SYSTEM HEALTH] {old_st} -> {new_st} (Reason: {reason})")
                    print(f"\n[INFO] End of video file '{input_source}'. Playback finished.")
                    break

                # Live camera temporary failure and reconnection recovery (Phase 9)
                max_reconnect = getattr(config, "CAMERA_MAX_RECONNECT_ATTEMPTS", 3)
                reconnect_delay = getattr(config, "CAMERA_RECONNECT_DELAY_SECONDS", 1.0)
                reconnect_enabled = getattr(config, "CAMERA_RECONNECT_ENABLED", True)

                if reconnect_enabled:
                    print(
                        f"\n[WARNING] Camera read failed. Attempting controlled recovery "
                        f"(max {max_reconnect} attempts)..."
                    )
                    reconnected = False
                    for attempt in range(1, max_reconnect + 1):
                        health_monitor.record_error(
                            "camera",
                            f"Camera stream disconnected, reconnect attempt {attempt}/{max_reconnect}",
                            fatal=False,
                            now=time.time(),
                        )
                        camera.release()
                        time.sleep(reconnect_delay)
                        print(
                            f"[CAMERA RECOVERY] Reconnect attempt {attempt}/{max_reconnect} on index {input_source}..."
                        )
                        new_cam = cv2.VideoCapture(input_source)
                        if new_cam.isOpened():
                            ok, test_frame = new_cam.read()
                            if ok:
                                camera = new_cam
                                frame = test_frame
                                success = True
                                reconnected = True
                                health_monitor.record_camera_open(True, source_desc, now=time.time())
                                health_monitor.record_frame_read(True, now=time.time())
                                print(f"[CAMERA RECOVERY SUCCESS] Camera reconnected on attempt {attempt}.")
                                break

                    if not reconnected:
                        health_monitor.record_camera_open(False, source_desc, now=time.time())
                        snap = health_monitor.evaluate(time.time())
                        transition = health_monitor.check_transition(snap)
                        if transition:
                            old_st, new_st, reason = transition
                            print(f"\n[SYSTEM HEALTH] {old_st} -> {new_st} (Reason: {reason})")
                        print(
                            f"\n[ERROR] Camera disconnected and recovery failed after {max_reconnect} attempts. Halting."
                        )
                        break
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

            # Periodic background queue draining (Phase 9 Offline-First)
            if frame_number % 30 == 0 and getattr(config, "OFFLINE_QUEUE_ENABLED", True):
                alert_q = get_alert_queue()
                if alert_q.pending_count > 0:

                    def _on_queue_drain(stats):
                        if stats.get("attempted", 0) > 0:
                            health_monitor.record_delivery_result(
                                success=(stats.get("sent", 0) > 0),
                                error=None if stats.get("failed", 0) == 0 else "Retry attempt failed",
                                pending_count=stats.get("pending_remaining", 0),
                            )
                        else:
                            health_monitor.update_pending_alerts(stats.get("pending_remaining", 0))

                    drain_delivery_queue_async(queue=alert_q, callback=_on_queue_drain)
                else:
                    health_monitor.update_pending_alerts(0)

            # Run YOLO detection & ByteTrack tracking (with recoverable inference failure handling)
            try:
                results = model.track(frame, persist=True, tracker=tracker_config, verbose=False)
                detector_consecutive_failures = 0
                health_monitor.record_detector_status(SUBSYSTEM_READY)
            except Exception as e:
                detector_consecutive_failures += 1
                detector_total_failures += 1
                max_det_fails = getattr(config, "DETECTOR_MAX_CONSECUTIVE_FAILURES", 3)
                is_fatal = detector_consecutive_failures >= max_det_fails
                det_status = SUBSYSTEM_FAILED if is_fatal else SUBSYSTEM_DEGRADED
                health_monitor.record_detector_status(
                    det_status,
                    f"Inference exception ({detector_consecutive_failures}/{max_det_fails}): {e}",
                    now=current_time,
                )
                print(f"\n[DETECTOR ERROR] Inference failure #{detector_consecutive_failures}: {e}")

                # SAFE FAILURE: Do NOT reset consecutive_elephant_frames to 0!
                # Do NOT invent an all-clear or false LOW risk!
                current_health = health_monitor.evaluate(current_time)
                hud_status = (
                    f"DETECTOR FAILED ({detector_consecutive_failures})"
                    if is_fatal
                    else f"DETECTOR DEGRADED ({detector_consecutive_failures})"
                )
                hud_color = config.COLOR_ALERT_RED if is_fatal else config.COLOR_WARN_YELLOW
                cooldown_remaining = max(0.0, config.ALERT_COOLDOWN_SECONDS - (current_time - last_alert_time))
                draw_hud(
                    frame=frame,
                    status_text=hud_status,
                    status_color=hud_color,
                    detection_count=consecutive_elephant_frames,
                    fps=fps,
                    cooldown_remaining=cooldown_remaining,
                    model_name=model_label,
                    tracking_summary=tracker.get_hud_summary(),
                    tracked_count=len(tracker.tracks),
                    risk_assessment=current_risk_assessment,
                    geo_mode="SIMULATION" if sim_mode else "OFF",
                    health_snapshot=current_health,
                )
                if not no_show:
                    cv2.imshow("Elephant Detection & Tracking - Early Warning System", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), ord("Q")]:
                        break

                if is_fatal:
                    print(
                        f"\n[CRITICAL] Detector failed {detector_consecutive_failures} consecutive times. "
                        f"Halting safely to prevent unmonitored operation."
                    )
                    break
                continue

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
            # 🗺️ GEOFENCING & INTELLIGENT THREAT ASSESSMENT 2.0
            # =================================================================
            elephant_found_in_frame = len(active_tracks) > 0
            max_elephant_confidence = max([t.confidence for t in active_tracks], default=0.0)

            if elephant_found_in_frame:
                consecutive_elephant_frames += 1
            else:
                consecutive_elephant_frames = 0

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
                    persistence_frames=consecutive_elephant_frames,
                    duration_seconds=event_manager.get_duration(current_time),
                    tracked_objects=[t.track_id for t in active_tracks],
                    event_id=event_manager.active_event_id,
                )

                transition = event_manager.update(
                    elephant_detected=True,
                    current_time=current_time,
                    risk_assessment=current_risk_assessment,
                )

                if transition.event_id and not current_risk_assessment.event_id:
                    current_risk_assessment.event_id = transition.event_id

                # Evaluate decoupled Alert Policy
                decision = alert_policy.evaluate(
                    assessment=current_risk_assessment,
                    current_time=current_time,
                    last_alert_time=last_alert_time,
                    last_alert_level=event_manager.last_alert_level,
                    is_escalated=transition.is_escalated,
                )

                if min_alert_risk and current_risk_assessment.level not in (min_alert_risk, RISK_CRITICAL, RISK_HIGH):
                    decision.should_dispatch = False

                if decision.should_dispatch:
                    tracked_alert_info = [
                        {"id": t.track_id, "conf": t.confidence, "movement": t.movement} for t in active_tracks
                    ]
                    risk_info_dict = current_risk_assessment.to_dict()
                    trigger_alert(
                        max_elephant_confidence,
                        tracked_info=tracked_alert_info,
                        risk_info=risk_info_dict,
                        health_monitor=health_monitor,
                        async_delivery=True,
                    )
                    event_manager.record_alert(current_risk_assessment.level, current_time)
                    last_alert_time = current_time
                    alert_banner_until = current_time + config.ALERT_BANNER_DURATION_SECONDS

            elif not elephant_found_in_frame:
                current_risk_assessment = None
                transition = event_manager.update(
                    elephant_detected=False,
                    current_time=current_time,
                )
                if transition.transition_type == "RESOLVED":
                    print(
                        f"\n[EVENT RESOLVED] Elephant left monitored scene. "
                        f"Event ID: {transition.resolved_event_id} | Total Duration: {transition.duration:.1f}s | "
                        f"Peak Risk: {transition.max_risk_level}"
                    )
                    resolved_event = create_alert_event(
                        confidence=0.0,
                        event_id=transition.resolved_event_id,
                        risk_info={
                            "risk_level": "LOW",
                            "risk_score": 0,
                            "event_state": "RESOLVED",
                            "is_resolved": True,
                            "resolution_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "duration_seconds": round(transition.duration, 1),
                        },
                    )
                    dispatch_alert(resolved_event, send_telegram=False, verbose=False)

            # Cooldown calculation for HUD
            time_since_last_alert = current_time - last_alert_time
            cooldown_active = time_since_last_alert < config.ALERT_COOLDOWN_SECONDS
            cooldown_remaining = (
                max(0.0, config.ALERT_COOLDOWN_SECONDS - time_since_last_alert) if cooldown_active else 0.0
            )

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

            # Update health monitoring with frame processing metrics
            health_monitor.record_frame_processed(
                detection_count=len(candidate_elephants),
                active_tracks=len(active_tracks),
                now=current_time,
            )
            current_health = health_monitor.evaluate(current_time)
            health_transition = health_monitor.check_transition(current_health)
            if health_transition:
                old_st, new_st, reason = health_transition
                print(f"\n[SYSTEM HEALTH] {old_st} -> {new_st} (Reason: {reason})")

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

            # Render HUD with health snapshot
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
                health_snapshot=current_health,
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
        health_monitor.record_error("pipeline", e, fatal=True)
        print(f"\nERROR: Unexpected error during execution: {e}")
        return False
    finally:
        print("[INFO] Releasing video source and closing windows...")
        camera.release()
        if not no_show:
            cv2.destroyAllWindows()
        final_health = health_monitor.get_snapshot()
        print(
            f"[INFO] Final System Health: {final_health.overall_status} "
            f"(Uptime: {final_health.uptime_seconds:.1f}s, "
            f"Processed: {final_health.processed_frames}/{final_health.total_frames} frames, "
            f"Errors: {final_health.error_count})"
        )
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
