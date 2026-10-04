"""
=============================================================================
🐘 PROJECT ZOGAN — PHASE 8 TEST SUITE: HEALTH MONITORING & OBSERVABILITY
=============================================================================

Comprehensive automated tests verifying:
  1. Healthy input
  2. No frame received (unknown state)
  3. Stale frame (freshness threshold)
  4. Camera open failure
  5. Single dropped frame
  6. Repeated frame failures
  7. Low FPS throughput
  8. FPS recovery
  9. Detector initialization failure
  10. Detector inference failure
  11. Tracker failure
  12. Risk engine failure
  13. No elephants detected (system remains HEALTHY)
  14. Elephant detected (threat risk is separate from system health)
  15. Healthy -> degraded transition
  16. Degraded -> healthy recovery
  17. Healthy -> offline transition
  18. Offline -> recovery
  19. Error counter & consecutive error tracking
  20. Error counter bounds (no memory leaks)
  21. Health snapshot serialization (.to_dict())
  22. Configuration overrides
  23. Uptime calculation & average FPS
=============================================================================
"""

import json
from monitoring import (
    HEALTH_DEGRADED,
    HEALTH_HEALTHY,
    HEALTH_OFFLINE,
    HEALTH_UNKNOWN,
    INPUT_DEGRADED,
    INPUT_OFFLINE,
    INPUT_ONLINE,
    INPUT_UNKNOWN,
    SUBSYSTEM_DEGRADED,
    SUBSYSTEM_FAILED,
    SUBSYSTEM_READY,
    SUBSYSTEM_UNKNOWN,
    FakeCamera,
    FakeClock,
    FaultyDetector,
    FaultyRiskEngine,
    FaultyTracker,
    SystemHealthMonitor,
)


def test_1_healthy_input():
    """Scenario 1: Input opened, fresh frames arriving at 30 FPS, subsystems ready."""
    clock = FakeClock(1000.0)
    camera = FakeCamera(opened=True)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(camera.isOpened(), source_info="webcam:0", now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())
    monitor.record_tracker_status(SUBSYSTEM_READY, now=clock.now())
    monitor.record_risk_engine_status(SUBSYSTEM_READY, now=clock.now())

    # Feed 15 frames at 30 FPS (delta = 0.033s)
    for _ in range(15):
        clock.advance(0.033)
        ret, _ = camera.read()
        monitor.record_frame_read(ret, now=clock.now())
        monitor.record_frame_processed(detection_count=0, active_tracks=0, now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.overall_status == HEALTH_HEALTHY
    assert snapshot.input_status == INPUT_ONLINE
    assert snapshot.detector_status == SUBSYSTEM_READY
    assert snapshot.tracker_status == SUBSYSTEM_READY
    assert snapshot.risk_engine_status == SUBSYSTEM_READY
    assert snapshot.current_fps > 25.0
    assert snapshot.consecutive_errors == 0
    assert "All monitored subsystems operational" in snapshot.reasons[0]


def test_2_no_frame_received():
    """Scenario 2: Camera opened, but zero frames have been received yet."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.overall_status == HEALTH_UNKNOWN
    assert snapshot.input_status == INPUT_UNKNOWN
    assert snapshot.detector_status == SUBSYSTEM_UNKNOWN
    assert snapshot.total_frames == 0
    assert snapshot.processed_frames == 0


def test_3_stale_frame():
    """Scenario 3: Normal feed halts; frame age exceeds offline threshold."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(
        freshness_degraded_seconds=1.5,
        freshness_offline_seconds=5.0,
        start_time=clock.now(),
    )
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    # Read one valid frame
    clock.advance(0.033)
    monitor.record_frame_read(True, now=clock.now())

    # Advance clock by 6.0 seconds (stream frozen / stalled)
    clock.advance(6.0)
    snapshot = monitor.evaluate(now=clock.now())

    assert snapshot.input_status == INPUT_OFFLINE
    assert snapshot.overall_status == HEALTH_OFFLINE
    assert any("offline" in r.lower() or "age" in r.lower() for r in snapshot.reasons)


def test_4_camera_open_failure():
    """Scenario 4: Camera stream fails to open initially."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(False, source_info="missing_camera", now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.overall_status == HEALTH_OFFLINE
    assert snapshot.input_status == INPUT_OFFLINE
    assert snapshot.error_count > 0
    assert any("failed to open" in r.lower() for r in snapshot.reasons)


def test_5_single_dropped_frame():
    """Scenario 5: Single dropped frame does not cause an immediate OFFLINE outage."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(
        max_consecutive_frame_failures=5,
        start_time=clock.now(),
    )
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    # Read several good frames
    for _ in range(5):
        clock.advance(0.033)
        monitor.record_frame_read(True, now=clock.now())

    # Single dropped frame
    clock.advance(0.033)
    monitor.record_frame_read(False, now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    # Should be DEGRADED due to consecutive_dropped_frames > 0, but NOT OFFLINE
    assert snapshot.overall_status == HEALTH_DEGRADED
    assert snapshot.input_status != INPUT_OFFLINE
    assert snapshot.dropped_frames == 1


def test_6_repeated_frame_failures():
    """Scenario 6: Multiple consecutive dropped frames trigger DEGRADED then OFFLINE."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(
        max_consecutive_frame_failures=3,
        max_consecutive_frame_failures_offline=6,
        start_time=clock.now(),
    )
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    # 1 valid frame first
    clock.advance(0.033)
    monitor.record_frame_read(True, now=clock.now())

    # 3 consecutive drops -> DEGRADED
    for _ in range(3):
        clock.advance(0.033)
        monitor.record_frame_read(False, now=clock.now())

    snap1 = monitor.evaluate(now=clock.now())
    assert snap1.input_status == INPUT_DEGRADED
    assert snap1.overall_status == HEALTH_DEGRADED

    # 3 more drops (total 6) -> OFFLINE
    for _ in range(3):
        clock.advance(0.033)
        monitor.record_frame_read(False, now=clock.now())

    snap2 = monitor.evaluate(now=clock.now())
    assert snap2.input_status == INPUT_OFFLINE
    assert snap2.overall_status == HEALTH_OFFLINE


def test_7_low_fps():
    """Scenario 7: Frames arrive too slowly, triggering DEGRADED status."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(
        fps_minimum=15.0,
        fps_window_size=10,
        start_time=clock.now(),
    )
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    # Feed frames at 5 FPS (delta = 0.2s)
    for _ in range(12):
        clock.advance(0.2)
        monitor.record_frame_read(True, now=clock.now())
        monitor.record_frame_processed(now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.current_fps < 10.0
    assert snapshot.overall_status == HEALTH_DEGRADED
    assert any("low fps" in r.lower() for r in snapshot.reasons)


def test_8_fps_recovery():
    """Scenario 8: System recovers from low FPS back to HEALTHY when throughput increases."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(
        fps_minimum=15.0,
        fps_window_size=10,
        start_time=clock.now(),
    )
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    # Slow period: 10 frames at 5 FPS
    for _ in range(10):
        clock.advance(0.2)
        monitor.record_frame_read(True, now=clock.now())
        monitor.record_frame_processed(now=clock.now())

    assert monitor.evaluate(now=clock.now()).overall_status == HEALTH_DEGRADED

    # Fast recovery: 15 frames at 30 FPS (delta = 0.033s)
    for _ in range(15):
        clock.advance(0.033)
        monitor.record_frame_read(True, now=clock.now())
        monitor.record_frame_processed(now=clock.now())

    recovered = monitor.evaluate(now=clock.now())
    assert recovered.current_fps > 25.0
    assert recovered.overall_status == HEALTH_HEALTHY


def test_9_detector_initialization_failure():
    """Scenario 9: Model weight load failure marks detector and system FAILED / OFFLINE."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())

    detector = FaultyDetector(fail_on_load=True)
    try:
        detector.load("missing.pt")
    except Exception as e:
        monitor.record_detector_status(SUBSYSTEM_FAILED, str(e), now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.detector_status == SUBSYSTEM_FAILED
    assert snapshot.overall_status == HEALTH_OFFLINE
    assert any("detector" in r.lower() for r in snapshot.reasons)


def test_10_detector_inference_failure():
    """Scenario 10: Non-fatal inference exception marks detector DEGRADED."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    # Simulated transient inference error
    monitor.record_detector_status(SUBSYSTEM_DEGRADED, "Transient CUDA error", now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.detector_status == SUBSYSTEM_DEGRADED
    assert snapshot.overall_status == HEALTH_DEGRADED


def test_11_tracker_failure():
    """Scenario 11: Tracker exception degrades tracker but keeps camera active."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    tracker = FaultyTracker(fail_on_update=True)
    try:
        tracker.update([])
    except Exception as e:
        monitor.record_tracker_status(SUBSYSTEM_FAILED, str(e), now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.tracker_status == SUBSYSTEM_FAILED
    assert snapshot.overall_status == HEALTH_DEGRADED
    assert any("tracking" in r.lower() for r in snapshot.reasons)


def test_12_risk_engine_failure():
    """Scenario 12: Risk engine exception degrades system gracefully."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())
    monitor.record_tracker_status(SUBSYSTEM_READY, now=clock.now())

    risk_engine = FaultyRiskEngine(fail_on_evaluate=True)
    try:
        risk_engine.evaluate()
    except Exception as e:
        monitor.record_risk_engine_status(SUBSYSTEM_FAILED, str(e), now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.risk_engine_status == SUBSYSTEM_FAILED
    assert snapshot.overall_status == HEALTH_DEGRADED
    assert any("risk" in r.lower() for r in snapshot.reasons)


def test_13_no_elephants_detected():
    """
    Scenario 13: Zero elephants detected over 100 frames.
    MANDATORY RULE: Zero elephants detected is NOT a detector or system failure.
    """
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())
    monitor.record_tracker_status(SUBSYSTEM_READY, now=clock.now())
    monitor.record_risk_engine_status(SUBSYSTEM_READY, now=clock.now())

    for _ in range(100):
        clock.advance(0.033)
        monitor.record_frame_read(True, now=clock.now())
        # Zero elephants detected!
        monitor.record_frame_processed(detection_count=0, active_tracks=0, now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.overall_status == HEALTH_HEALTHY
    assert snapshot.detector_status == SUBSYSTEM_READY
    assert snapshot.processed_frames == 100
    assert snapshot.error_count == 0


def test_14_elephant_detected_risk_isolation():
    """
    Scenario 14: Elephant is detected and threat risk is CRITICAL.
    MANDATORY RULE: Threat risk level is isolated from system health.
    """
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())
    monitor.record_tracker_status(SUBSYSTEM_READY, now=clock.now())
    monitor.record_risk_engine_status(SUBSYSTEM_READY, now=clock.now())

    for _ in range(20):
        clock.advance(0.033)
        monitor.record_frame_read(True, now=clock.now())
        # Elephant detected in every frame
        monitor.record_frame_processed(detection_count=2, active_tracks=2, now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert snapshot.overall_status == HEALTH_HEALTHY
    assert snapshot.total_frames == 20


def test_15_healthy_to_degraded_transition():
    """Scenario 15: State transition listener detects HEALTHY -> DEGRADED."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    # Establish HEALTHY
    clock.advance(0.033)
    monitor.record_frame_read(True, now=clock.now())
    monitor.record_frame_processed(now=clock.now())
    snap1 = monitor.evaluate(now=clock.now())
    assert monitor.check_transition(snap1) is None  # Initial baseline

    # Drop frames to trigger DEGRADED
    for _ in range(5):
        clock.advance(0.033)
        monitor.record_frame_read(False, now=clock.now())

    snap2 = monitor.evaluate(now=clock.now())
    trans = monitor.check_transition(snap2)
    assert trans is not None
    old_st, new_st, reason = trans
    assert old_st == HEALTH_HEALTHY
    assert new_st == HEALTH_DEGRADED


def test_16_degraded_to_healthy_recovery():
    """Scenario 16: State transition listener detects DEGRADED -> HEALTHY recovery."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    # Force DEGRADED
    monitor.record_tracker_status(SUBSYSTEM_DEGRADED, "Temporary track swap", now=clock.now())
    snap1 = monitor.evaluate(now=clock.now())
    monitor.check_transition(snap1)
    assert snap1.overall_status == HEALTH_DEGRADED

    # Recover tracker
    monitor.record_tracker_status(SUBSYSTEM_READY, now=clock.now())
    clock.advance(0.033)
    monitor.record_frame_read(True, now=clock.now())
    monitor.record_frame_processed(now=clock.now())

    snap2 = monitor.evaluate(now=clock.now())
    trans = monitor.check_transition(snap2)
    assert trans is not None
    old_st, new_st, reason = trans
    assert old_st == HEALTH_DEGRADED
    assert new_st == HEALTH_HEALTHY


def test_17_healthy_to_offline():
    """Scenario 17: Sudden complete camera loss transitions to OFFLINE."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(
        freshness_offline_seconds=3.0,
        start_time=clock.now(),
    )
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())

    # 1 valid frame
    clock.advance(0.033)
    monitor.record_frame_read(True, now=clock.now())
    monitor.record_frame_processed(now=clock.now())
    snap1 = monitor.evaluate(now=clock.now())
    monitor.check_transition(snap1)

    # 4 seconds pass with zero frames
    clock.advance(4.0)
    snap2 = monitor.evaluate(now=clock.now())
    trans = monitor.check_transition(snap2)
    assert trans is not None
    assert trans[0] == HEALTH_HEALTHY
    assert trans[1] == HEALTH_OFFLINE


def test_18_offline_to_recovery():
    """Scenario 18: Stream recovers after being offline."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(
        freshness_offline_seconds=3.0,
        start_time=clock.now(),
    )
    monitor.record_camera_open(True, now=clock.now())
    # 1 valid frame initially
    clock.advance(0.033)
    monitor.record_frame_read(True, now=clock.now())
    monitor.record_frame_processed(now=clock.now())

    # Offline state: 5.0 seconds elapse without frames
    clock.advance(5.0)
    snap1 = monitor.evaluate(now=clock.now())
    monitor.check_transition(snap1)
    assert snap1.overall_status == HEALTH_OFFLINE

    # Reconnection & fresh frames
    for _ in range(5):
        clock.advance(0.033)
        monitor.record_frame_read(True, now=clock.now())
        monitor.record_frame_processed(now=clock.now())

    snap2 = monitor.evaluate(now=clock.now())
    trans = monitor.check_transition(snap2)
    assert trans is not None
    assert trans[0] == HEALTH_OFFLINE
    assert trans[1] == HEALTH_HEALTHY


def test_19_error_counter():
    """Scenario 19: Error recording increments count and decays on clean processing."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())

    monitor.record_error("pipeline", "Disk full warning", now=clock.now())
    monitor.record_error("pipeline", "IO timeout", now=clock.now())

    assert monitor._metrics.error_count == 2
    assert monitor._metrics.consecutive_errors == 2

    # Successful processing decrements consecutive errors
    monitor.record_frame_processed(now=clock.now())
    assert monitor._metrics.consecutive_errors == 1


def test_20_error_counter_bounds():
    """Scenario 20: Memory bounds on error history prevent unbounded growth."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(max_error_history=10, start_time=clock.now())

    for i in range(50):
        monitor.record_error("subsystem", f"Error event #{i}", now=clock.now())

    assert monitor._metrics.error_count == 50
    assert len(monitor._error_history) == 10  # Bounded queue


def test_21_health_snapshot_serialization():
    """Scenario 21: HealthSnapshot.to_dict() produces clean JSON schema."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())
    monitor.record_camera_open(True, now=clock.now())
    monitor.record_detector_status(SUBSYSTEM_READY, now=clock.now())
    clock.advance(0.033)
    monitor.record_frame_read(True, now=clock.now())
    monitor.record_frame_processed(detection_count=1, active_tracks=1, now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    snap_dict = snapshot.to_dict()

    # Must be valid JSON
    serialized = json.dumps(snap_dict)
    assert isinstance(serialized, str)

    expected_keys = {
        "overall_status",
        "input_status",
        "detector_status",
        "tracker_status",
        "risk_engine_status",
        "current_fps",
        "average_fps",
        "last_frame_age_seconds",
        "uptime_seconds",
        "total_frames",
        "processed_frames",
        "dropped_frames",
        "error_count",
        "consecutive_errors",
        "last_error",
        "last_error_time",
        "reasons",
        "timestamp",
    }
    assert expected_keys.issubset(set(snap_dict.keys()))


def test_22_configuration_overrides():
    """Scenario 22: Custom initialization thresholds take effect cleanly."""
    monitor = SystemHealthMonitor(
        fps_minimum=24.0,
        fps_window_size=45,
        freshness_degraded_seconds=0.8,
        freshness_offline_seconds=2.5,
        max_consecutive_frame_failures=8,
    )
    assert monitor.fps_minimum == 24.0
    assert monitor.fps_window_size == 45
    assert monitor.freshness_degraded_seconds == 0.8
    assert monitor.freshness_offline_seconds == 2.5
    assert monitor.max_consecutive_frame_failures == 8


def test_23_uptime_and_average_fps():
    """Scenario 23: Accurate uptime and throughput calculations."""
    clock = FakeClock(1000.0)
    monitor = SystemHealthMonitor(start_time=clock.now())

    # Advance 10 seconds and process 300 frames
    clock.advance(10.0)
    for _ in range(300):
        monitor.record_frame_processed(now=clock.now())

    snapshot = monitor.evaluate(now=clock.now())
    assert abs(snapshot.uptime_seconds - 10.0) < 0.01
    assert abs(snapshot.average_fps - 30.0) < 0.01
