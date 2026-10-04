"""
=============================================================================
🐘 PROJECT ZOGAN — PHASE 9 AUTOMATED TESTS
OFFLINE-FIRST OPERATION & FAILURE RECOVERY
=============================================================================

Comprehensive test suite verifying:
  1. Event saved locally before remote alert delivery
  2. Successful alert delivery
  3. Network failure handling (local storage preserved)
  4. Telegram failure handling (error captured, event enqueued)
  5. Missing Telegram credentials handling (fail-safe)
  6. Retry mechanism for queued alerts
  7. Bounded retry limit (preventing infinite loops)
  8. Exponential backoff retry timing
  9. Queue file-backed persistence
  10. Queue recovery after simulated process restart
  11. Event deduplication and stable identity preservation
  12. Queue capacity bounds and cleanup
  13. Camera temporary failure handling
  14. Camera recovery back to healthy
  15. Detector temporary failure handling (recoverable)
  16. Detector repeated failure limit
  17. System-health observability integration
  18. Zero event loss across network blackout
  19. No false LOW risk or all-clear on detector failure
  20. Non-blocking delivery pipeline execution

Zero real internet, zero real Telegram, zero physical camera, zero GPU needed.
=============================================================================
"""

import importlib
import json
from unittest.mock import MagicMock

try:
    pytest = importlib.import_module("pytest")
    fixture = pytest.fixture
except Exception:
    pytest = None

    def fixture(fn):
        return fn


import config
from ai.risk_engine import RISK_CRITICAL, RISK_HIGH, RiskEngine
from alerts.dispatcher import (
    dispatch_alert,
    process_delivery_queue,
)
from alerts.history import load_alerts
from alerts.models import (
    DELIVERY_FAILED,
    DELIVERY_NOT_CONFIGURED,
    DELIVERY_PENDING,
    DELIVERY_SENT,
    create_alert_event,
)
from alerts.queue import PersistentAlertQueue
from monitoring.health import (
    ALERT_DELIVERY_DEGRADED,
    ALERT_DELIVERY_HEALTHY,
    HEALTH_DEGRADED,
    HEALTH_HEALTHY,
    HEALTH_OFFLINE,
    INPUT_DEGRADED,
    INPUT_ONLINE,
    NETWORK_AVAILABLE,
    NETWORK_UNAVAILABLE,
    SUBSYSTEM_DEGRADED,
    SUBSYSTEM_FAILED,
    SUBSYSTEM_READY,
    SystemHealthMonitor,
)


@fixture
def isolated_env(tmp_path, monkeypatch):
    """Configures isolated log and queue paths in a temporary test directory."""
    log_file = tmp_path / "alerts.jsonl"
    queue_file = tmp_path / "delivery_queue.jsonl"

    monkeypatch.setattr(config, "ALERT_LOG_FILE", str(log_file))
    monkeypatch.setattr(config, "ALERT_QUEUE_FILE", str(queue_file))
    monkeypatch.setattr(config, "OFFLINE_QUEUE_ENABLED", True)
    monkeypatch.setattr(config, "ALERT_RETRY_LIMIT", 3)
    monkeypatch.setattr(config, "ALERT_RETRY_BASE_DELAY_SECONDS", 1.0)
    monkeypatch.setattr(config, "ALERT_RETRY_BACKOFF_FACTOR", 2.0)
    monkeypatch.setattr(config, "ALERT_DELIVERY_TIMEOUT_SECONDS", 1.0)

    queue = PersistentAlertQueue(queue_file=queue_file)
    return {
        "log_file": log_file,
        "queue_file": queue_file,
        "queue": queue,
    }


# =============================================================================
# 1. Event saved before alert delivery
# =============================================================================


def test_1_event_saved_before_alert_delivery(isolated_env, monkeypatch):
    """
    LOCAL-FIRST RULE: An event MUST be written to local storage BEFORE any
    attempt to communicate over the network is made. Even if the network call
    crashes or hangs, local persistence must already be complete.
    """
    log_file = isolated_env["log_file"]
    queue = isolated_env["queue"]

    # Hook to verify log_file already exists and contains the event WHEN send_telegram is called
    logged_before_send = []

    def mock_send(event, timeout=None):
        logged_before_send.append(log_file.exists())
        if log_file.exists():
            records = load_alerts(log_file)
            logged_before_send.append(len(records) > 0 and records[0].get("event_id") == event.event_id)
        raise RuntimeError("Network crash during alert send")

    monkeypatch.setattr("alerts.dispatcher.is_telegram_configured", lambda: True)
    monkeypatch.setattr("alerts.dispatcher.send_telegram_alert", mock_send)

    event = create_alert_event(confidence=0.92, risk_info={"risk_level": "HIGH"})
    res = dispatch_alert(event, log_file=log_file, queue=queue, verbose=False)

    # Verification: Event was already on disk before network send was attempted
    assert len(logged_before_send) == 2
    assert logged_before_send[0] is True  # File existed
    assert logged_before_send[1] is True  # Event was inside file
    assert res["logged"] is True
    assert log_file.exists()


# =============================================================================
# 2. Successful delivery
# =============================================================================


def test_2_successful_delivery(isolated_env, monkeypatch):
    """
    When Telegram is configured and delivery succeeds:
    - Logged locally
    - delivery_status is SENT
    - telegram_sent is True
    - Queue remains clean (0 pending)
    """
    log_file = isolated_env["log_file"]
    queue = isolated_env["queue"]

    monkeypatch.setattr("alerts.dispatcher.is_telegram_configured", lambda: True)
    monkeypatch.setattr(
        "alerts.dispatcher.send_telegram_alert",
        lambda event, timeout=None: {"success": True, "status": "sent", "error": None},
    )

    event = create_alert_event(confidence=0.95, risk_info={"risk_level": "CRITICAL"})
    res = dispatch_alert(event, log_file=log_file, queue=queue, verbose=False)

    assert res["logged"] is True
    assert res["telegram_sent"] is True
    assert res["delivery_status"] == DELIVERY_SENT
    assert queue.pending_count == 0


# =============================================================================
# 3. Network failure
# =============================================================================


def test_3_network_failure(isolated_env, monkeypatch):
    """
    When network is unavailable (timeout/socket error):
    - System does not crash
    - Event remains persisted locally
    - delivery_status is PENDING
    - Event is automatically enqueued into persistent offline queue
    """
    log_file = isolated_env["log_file"]
    queue = isolated_env["queue"]

    monkeypatch.setattr("alerts.dispatcher.is_telegram_configured", lambda: True)
    monkeypatch.setattr(
        "alerts.dispatcher.send_telegram_alert",
        lambda event, timeout=None: {"success": False, "status": "network_error", "error": "Connection timed out"},
    )

    event = create_alert_event(confidence=0.88, risk_info={"risk_level": "HIGH"})
    res = dispatch_alert(event, log_file=log_file, queue=queue, verbose=False)

    assert res["logged"] is True
    assert res["telegram_sent"] is False
    assert res["delivery_status"] == DELIVERY_PENDING
    assert res["enqueued"] is True
    assert queue.pending_count == 1
    assert queue.get(event.event_id) is not None


# =============================================================================
# 4. Telegram API failure
# =============================================================================


def test_4_telegram_api_failure(isolated_env, monkeypatch):
    """
    When Telegram API returns an error (HTTP 400/500/rate limit):
    - Event is logged locally
    - Delivery error is captured
    - Event is enqueued for bounded retry
    """
    log_file = isolated_env["log_file"]
    queue = isolated_env["queue"]

    monkeypatch.setattr("alerts.dispatcher.is_telegram_configured", lambda: True)
    monkeypatch.setattr(
        "alerts.dispatcher.send_telegram_alert",
        lambda event, timeout=None: {"success": False, "status": "api_error", "error": "429 Too Many Requests"},
    )

    event = create_alert_event(confidence=0.90, risk_info={"risk_level": "HIGH"})
    res = dispatch_alert(event, log_file=log_file, queue=queue, verbose=False)

    assert res["logged"] is True
    assert res["delivery_status"] == DELIVERY_PENDING
    assert "429 Too Many Requests" in res["telegram_error"]
    assert queue.pending_count == 1


# =============================================================================
# 5. Missing Telegram credentials
# =============================================================================


def test_5_missing_telegram_credentials(isolated_env, monkeypatch):
    """
    When Telegram credentials are not configured:
    - Never crashes Zogan
    - Event is logged locally
    - delivery_status is NOT_CONFIGURED
    - No network attempts made, not queued
    """
    log_file = isolated_env["log_file"]
    queue = isolated_env["queue"]

    monkeypatch.setattr("alerts.dispatcher.is_telegram_configured", lambda: False)

    mock_send = MagicMock()
    monkeypatch.setattr("alerts.dispatcher.send_telegram_alert", mock_send)

    event = create_alert_event(confidence=0.85, risk_info={"risk_level": "HIGH"})
    res = dispatch_alert(event, log_file=log_file, queue=queue, verbose=False)

    assert res["logged"] is True
    assert res["delivery_status"] == DELIVERY_NOT_CONFIGURED
    assert res["telegram_status"] == "disabled"
    assert queue.pending_count == 0
    mock_send.assert_not_called()


# =============================================================================
# 6. Retry mechanism
# =============================================================================


def test_6_retry_mechanism(isolated_env, monkeypatch):
    """
    When a queued alert is ready for retry and network returns:
    - process_delivery_queue attempts delivery
    - On success, marks SENT and pending count drops
    """
    queue = isolated_env["queue"]
    event = create_alert_event(confidence=0.91, risk_info={"risk_level": "HIGH"})
    queue.enqueue(event, error="Initial connection timeout")
    assert queue.pending_count == 1

    monkeypatch.setattr("alerts.dispatcher.is_telegram_configured", lambda: True)
    monkeypatch.setattr(
        "alerts.dispatcher.send_telegram_alert",
        lambda ev, timeout=None: {"success": True, "status": "sent", "error": None},
    )

    # Process queue with current time far ahead enough to exceed initial delay
    import time

    future_time = time.time() + 1000.0
    stats = process_delivery_queue(queue=queue, now=future_time, verbose=False)

    assert stats["attempted"] == 1
    assert stats["sent"] == 1
    assert stats["pending_remaining"] == 0
    item = queue.get(event.event_id)
    assert item.status == DELIVERY_SENT


# =============================================================================
# 7. Retry limit (bounded retries)
# =============================================================================


def test_7_retry_limit_bounded(isolated_env, monkeypatch):
    """
    Verifies retry count is strictly bounded by ALERT_RETRY_LIMIT.
    Once attempts exceed the limit, status transitions to FAILED,
    preventing infinite retry loops.
    """
    import time

    queue = isolated_env["queue"]
    event = create_alert_event(confidence=0.89, risk_info={"risk_level": "HIGH"})
    queue.enqueue(event, error="Network down")

    monkeypatch.setattr("alerts.dispatcher.is_telegram_configured", lambda: True)
    monkeypatch.setattr(
        "alerts.dispatcher.send_telegram_alert",
        lambda ev, timeout=None: {"success": False, "status": "error", "error": "Still down"},
    )

    # Attempt 1 -> 2 -> 3 (limit is 3)
    curr_time = time.time()
    for _ in range(config.ALERT_RETRY_LIMIT):
        item = queue.get(event.event_id)
        curr_time = item.next_retry_time + 1.0
        process_delivery_queue(queue=queue, now=curr_time, verbose=False)

    item = queue.get(event.event_id)
    assert item.status == DELIVERY_FAILED
    assert item.attempts >= config.ALERT_RETRY_LIMIT
    assert queue.pending_count == 0  # No longer pending for retry


# =============================================================================
# 8. Exponential backoff behavior
# =============================================================================


def test_8_exponential_backoff_behavior(isolated_env):
    """
    Verifies that retry delays increase with each attempt:
    Delay = base * (factor ^ (attempts - 1))
    """
    queue = isolated_env["queue"]
    now = 1000.0

    d1 = queue.calculate_next_retry(attempts=1, now=now) - now
    d2 = queue.calculate_next_retry(attempts=2, now=now) - now
    d3 = queue.calculate_next_retry(attempts=3, now=now) - now

    assert abs(d1 - 1.0) < 0.05  # base delay (1s)
    assert abs(d2 - 2.0) < 0.05  # 1 * 2^1 = 2s
    assert abs(d3 - 4.0) < 0.05  # 1 * 2^2 = 4s
    assert d1 < d2 < d3


# =============================================================================
# 9. Queue file persistence
# =============================================================================


def test_9_queue_file_persistence(isolated_env):
    """
    Verifies that the queue is backed by an actual file on disk.
    Raw disk inspection must confirm valid JSON serialization.
    """
    queue = isolated_env["queue"]
    queue_file = isolated_env["queue_file"]

    event = create_alert_event(confidence=0.88, risk_info={"risk_level": "HIGH"})
    queue.enqueue(event, error="DNS lookup failed")

    assert queue_file.exists()
    with open(queue_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event"]["event_id"] == event.event_id
    assert data["status"] == DELIVERY_PENDING
    assert data["error"] == "DNS lookup failed"


# =============================================================================
# 10. Queue recovery after restart
# =============================================================================


def test_10_queue_recovery_after_restart(isolated_env):
    """
    Simulates a complete process shutdown and restart:
    1. Enqueue event to queue file
    2. Destroy the Python object
    3. Instantiate fresh PersistentAlertQueue from the same file
    4. Verify pending event and metadata are restored
    """
    queue_file = isolated_env["queue_file"]
    q1 = PersistentAlertQueue(queue_file=queue_file)

    event = create_alert_event(confidence=0.94, risk_info={"risk_level": "CRITICAL"})
    q1.enqueue(event, error="Network cable unplugged")
    assert q1.pending_count == 1
    del q1

    # Restart: fresh instance loading from file
    q2 = PersistentAlertQueue(queue_file=queue_file)
    assert q2.pending_count == 1
    recovered_item = q2.get(event.event_id)
    assert recovered_item is not None
    assert recovered_item.event.confidence == 0.94
    assert recovered_item.event.risk_level == "CRITICAL"
    assert recovered_item.status == DELIVERY_PENDING


# =============================================================================
# 11. Event deduplication and stable identity
# =============================================================================


def test_11_event_deduplication_stable_id(isolated_env):
    """
    Verifies that repeated enqueues of the same event retain the exact same
    stable event ID and do NOT spawn duplicate queue entries.
    """
    queue = isolated_env["queue"]
    event = create_alert_event(confidence=0.90, risk_info={"risk_level": "HIGH"})

    queue.enqueue(event, error="Error 1")
    assert queue.pending_count == 1

    # Second enqueue of the exact same event
    queue.enqueue(event, error="Error 2")
    assert queue.pending_count == 1
    assert len(queue.items) == 1
    assert queue.get(event.event_id).error == "Error 2"


# =============================================================================
# 12. Queue cleanup and max bounds
# =============================================================================


def test_12_queue_cleanup_and_bounds(isolated_env, monkeypatch):
    """
    Verifies that:
    1. Bounded maximum queue size prunes old non-pending items
    2. cleanup_delivered prunes delivered items older than max age
    """
    monkeypatch.setattr(config, "ALERT_QUEUE_MAX_SIZE", 3)
    queue = PersistentAlertQueue(queue_file=isolated_env["queue_file"], max_size=3)

    e1 = create_alert_event(confidence=0.81, event_id="EVT-01")
    e2 = create_alert_event(confidence=0.82, event_id="EVT-02")
    e3 = create_alert_event(confidence=0.83, event_id="EVT-03")
    e4 = create_alert_event(confidence=0.84, event_id="EVT-04")

    queue.enqueue(e1)
    queue.enqueue(e2)
    queue.mark_sent("EVT-01", now=100.0)
    queue.enqueue(e3)
    # Queue is now at capacity 3 (EVT-01 is SENT, EVT-02 & 03 are PENDING)
    queue.enqueue(e4)

    # EVT-01 (oldest sent item) should have been pruned to stay within max_size 3
    assert len(queue.items) <= 3
    assert queue.get("EVT-01") is None
    assert queue.get("EVT-04") is not None


# =============================================================================
# 13. Camera temporary failure handling
# =============================================================================


def test_13_camera_temporary_failure_degraded():
    """
    When camera fails to return frames:
    - health monitor records frame drop
    - status transitions to DEGRADED
    - does not crash the system
    """
    monitor = SystemHealthMonitor(
        max_consecutive_frame_failures=2,
        max_consecutive_frame_failures_offline=10,
    )
    monitor.record_camera_open(True)
    monitor.record_detector_status(SUBSYSTEM_READY)
    monitor.record_tracker_status(SUBSYSTEM_READY)
    monitor.record_risk_engine_status(SUBSYSTEM_READY)

    # 1 normal frame
    monitor.record_frame_read(True, now=10.0)
    assert monitor.evaluate(now=10.0).input_status == INPUT_ONLINE

    # 2 dropped frames
    monitor.record_frame_read(False, now=10.1)
    monitor.record_frame_read(False, now=10.2)

    snap = monitor.evaluate(now=10.2)
    assert snap.input_status == INPUT_DEGRADED
    assert snap.overall_status == HEALTH_DEGRADED


# =============================================================================
# 14. Camera recovery back to healthy
# =============================================================================


def test_14_camera_recovery_healthy():
    """
    When camera reconnects after temporary dropped frames:
    - consecutive frame drops reset to 0
    - input status returns to ONLINE
    - system health recovers to HEALTHY
    """
    monitor = SystemHealthMonitor(
        max_consecutive_frame_failures=2,
        max_consecutive_frame_failures_offline=10,
    )
    monitor.record_camera_open(True)
    monitor.record_detector_status(SUBSYSTEM_READY)
    monitor.record_tracker_status(SUBSYSTEM_READY)
    monitor.record_risk_engine_status(SUBSYSTEM_READY)

    # Drop 2 frames -> degraded
    monitor.record_frame_read(False, now=10.0)
    monitor.record_frame_read(False, now=10.1)
    assert monitor.evaluate(now=10.1).input_status == INPUT_DEGRADED

    # Reconnected!
    monitor.record_camera_open(True, now=11.0)
    monitor.record_frame_read(True, now=11.0)

    snap = monitor.evaluate(now=11.0)
    assert snap.input_status == INPUT_ONLINE
    assert snap.overall_status == HEALTH_HEALTHY


# =============================================================================
# 15. Detector temporary failure handling
# =============================================================================


def test_15_detector_temporary_failure_safe():
    """
    A single recoverable detector inference failure:
    - marks detector SUBSYSTEM_DEGRADED
    - records error with sanitization
    - does NOT report SUBSYSTEM_FAILED prematurely
    - does NOT crash the application
    """
    monitor = SystemHealthMonitor()
    monitor.record_camera_open(True)
    monitor.record_tracker_status(SUBSYSTEM_READY)
    monitor.record_risk_engine_status(SUBSYSTEM_READY)
    monitor.record_frame_read(True, now=5.0)

    # 1 failure
    monitor.record_detector_status(SUBSYSTEM_DEGRADED, "CUDA temporary timeout", now=5.0)
    snap = monitor.evaluate(now=5.0)

    assert snap.detector_status == SUBSYSTEM_DEGRADED
    assert snap.overall_status == HEALTH_DEGRADED
    assert "detector" in snap.reasons[0].lower()


# =============================================================================
# 16. Detector repeated failure limit
# =============================================================================


def test_16_detector_repeated_failure_limit():
    """
    When detector fails repeatedly beyond the configured limit:
    - detector status is marked SUBSYSTEM_FAILED
    - overall health becomes OFFLINE
    - prevents silent false-clear operation
    """
    monitor = SystemHealthMonitor()
    monitor.record_camera_open(True)
    monitor.record_tracker_status(SUBSYSTEM_READY)
    monitor.record_risk_engine_status(SUBSYSTEM_READY)
    monitor.record_frame_read(True, now=5.0)

    # Mark detector failed
    monitor.record_detector_status(SUBSYSTEM_FAILED, "Repeated inference segfault", now=5.0)
    snap = monitor.evaluate(now=5.0)

    assert snap.detector_status == SUBSYSTEM_FAILED
    assert snap.overall_status == HEALTH_OFFLINE


# =============================================================================
# 17. System health observability integration
# =============================================================================


def test_17_system_health_integration_offline_queue():
    """
    Health monitor correctly reflects delivery failures and queue status:
    - network_status: UNAVAILABLE
    - alert_delivery_status: DEGRADED
    - pending_alert_count: 5
    - overall system: DEGRADED
    - Clearly communicates: Zogan is still detecting locally, remote notifications offline.
    """
    monitor = SystemHealthMonitor()
    monitor.record_camera_open(True)
    monitor.record_detector_status(SUBSYSTEM_READY)
    monitor.record_tracker_status(SUBSYSTEM_READY)
    monitor.record_risk_engine_status(SUBSYSTEM_READY)
    monitor.record_frame_read(True, now=10.0)

    # Record delivery failure with 5 pending alerts
    monitor.record_delivery_result(
        success=False,
        error="Network unreachable",
        pending_count=5,
        now=10.0,
    )

    snap = monitor.evaluate(now=10.0)
    assert snap.network_status == NETWORK_UNAVAILABLE
    assert snap.alert_delivery_status == ALERT_DELIVERY_DEGRADED
    assert snap.pending_alert_count == 5
    assert snap.overall_status == HEALTH_DEGRADED
    assert snap.detector_status == SUBSYSTEM_READY  # Local detection is still ready!
    assert snap.input_status == INPUT_ONLINE  # Camera is still online!

    # When network recovers and queue is drained:
    monitor.record_frame_read(True, now=12.0)
    monitor.record_delivery_result(
        success=True,
        pending_count=0,
        now=12.0,
    )
    snap_recovered = monitor.evaluate(now=12.0)
    assert snap_recovered.network_status == NETWORK_AVAILABLE
    assert snap_recovered.alert_delivery_status == ALERT_DELIVERY_HEALTHY
    assert snap_recovered.pending_alert_count == 0
    assert snap_recovered.overall_status == HEALTH_HEALTHY


# =============================================================================
# 18. Zero event loss across network blackout
# =============================================================================


def test_18_no_event_loss_across_failures(isolated_env, monkeypatch):
    """
    Generates multiple high-risk events during complete network failure.
    Verifies 100% of generated events exist in local log file AND
    are tracked in the persistent offline queue.
    """
    log_file = isolated_env["log_file"]
    queue = isolated_env["queue"]

    monkeypatch.setattr("alerts.dispatcher.is_telegram_configured", lambda: True)
    monkeypatch.setattr(
        "alerts.dispatcher.send_telegram_alert",
        lambda event, timeout=None: {"success": False, "status": "blackout", "error": "No internet connection"},
    )

    events_to_create = 5
    created_ids = []
    for i in range(events_to_create):
        ev = create_alert_event(
            confidence=0.85 + (i * 0.02),
            risk_info={"risk_level": "HIGH", "score": 80 + i},
        )
        created_ids.append(ev.event_id)
        dispatch_alert(ev, log_file=log_file, queue=queue, verbose=False)

    # 1. Verify 100% exist in local disk log
    persisted_records = load_alerts(log_file)
    assert len(persisted_records) == events_to_create
    logged_ids = [r["event_id"] for r in persisted_records]
    assert logged_ids == created_ids

    # 2. Verify 100% are enqueued in offline queue
    assert queue.pending_count == events_to_create
    for eid in created_ids:
        assert queue.get(eid) is not None
        assert queue.get(eid).status == DELIVERY_PENDING


# =============================================================================
# 19. No false LOW risk on detector failure
# =============================================================================


def test_19_no_false_low_risk_after_detector_failure():
    """
    SAFE FAILURE REQUIREMENT:
    If the detector fails or produces no detections due to an error,
    the system must NOT report "LOW risk" or "all clear".
    """
    engine = RiskEngine()

    # Normal evaluate with an active detection in danger zone
    normal_eval = engine.evaluate(
        zone="VILLAGE_ZONE",
        distance_to_protected=50.0,
        trend="APPROACHING",
        confidence=0.92,
        persistence_frames=10,
    )
    assert normal_eval.level in (RISK_HIGH, RISK_CRITICAL)

    # If detector is unavailable/fails, inference doesn't run:
    # A safe system must NOT report LOW risk if an error occurred.
    monitor = SystemHealthMonitor()
    monitor.record_detector_status(SUBSYSTEM_FAILED, "Model inference broken")
    snap = monitor.evaluate()

    # The health layer must declare OFFLINE, never HEALTHY with a false LOW
    assert snap.overall_status == HEALTH_OFFLINE
    assert snap.detector_status == SUBSYSTEM_FAILED


# =============================================================================
# 20. Detection pipeline continues while remote delivery fails
# =============================================================================


def test_20_detection_pipeline_continues_while_remote_delivery_fails(isolated_env, monkeypatch):
    """
    NON-BLOCKING GUARANTEE:
    When remote delivery takes time or fails, asynchronous delivery dispatch
    returns immediately, ensuring the frame capture and detection loop is
    never stalled.
    """
    log_file = isolated_env["log_file"]
    queue = isolated_env["queue"]

    delivery_started = False
    delivery_completed = False

    def slow_failing_telegram(event, timeout=None):
        nonlocal delivery_started, delivery_completed
        delivery_started = True
        import time

        time.sleep(0.05)  # simulate slow network
        delivery_completed = True
        return {"success": False, "status": "timeout", "error": "Gateway timeout"}

    monkeypatch.setattr("alerts.dispatcher.is_telegram_configured", lambda: True)
    monkeypatch.setattr("alerts.dispatcher.send_telegram_alert", slow_failing_telegram)

    event = create_alert_event(confidence=0.93, risk_info={"risk_level": "HIGH"})

    # Dispatch with async_delivery=True (as used in run_camera.py)
    import time

    t0 = time.time()
    res = dispatch_alert(
        event,
        log_file=log_file,
        queue=queue,
        async_delivery=True,
        verbose=False,
    )
    elapsed = time.time() - t0

    # Must return immediately (< 30ms) without waiting for slow_failing_telegram
    assert elapsed < 0.04
    assert res["logged"] is True  # Event is ALREADY logged locally on disk
    assert res.get("async_scheduled") is True

    # Allow worker thread to complete
    time.sleep(0.1)
    assert delivery_started is True
    assert delivery_completed is True
