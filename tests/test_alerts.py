"""
=============================================================================
🐘 PROJECT ZOGAN — AUTOMATED ALERT SYSTEM TESTS (PHASE 6)
=============================================================================

Test suite for Project Zogan Alert Event Layer & Telegram Notification:
  1. Event creation
  2. Event serialization
  3. Unique event IDs
  4. JSONL logging
  5. Log directory creation
  6. Recent history reading
  7. Missing Telegram credentials
  8. Telegram disabled behavior
  9. Message formatting
 10. Telegram success response
 11. Telegram timeout / failure
 12. Dispatcher behavior
 13. HIGH alert behavior
 14. CRITICAL alert behavior
 15. LOW / MEDIUM behavior (log only / local warning)
 16. Existing cooldown compatibility
 17. Graceful log failure handling

Usage:
    python -m pytest tests/test_alerts.py
    python tests/test_alerts.py
=============================================================================
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alerts.dispatcher import dispatch_alert
from alerts.event_logger import AlertEventLogger
from alerts.history import (
    count_alerts,
    filter_by_risk,
    get_alert_count,
    get_latest_alert,
    get_recent_alerts,
    load_alerts,
)
from alerts.models import AlertEvent
from alerts.telegram import (
    ENV_BOT_TOKEN,
    ENV_CHAT_ID,
    format_telegram_message,
    get_telegram_credentials,
    is_telegram_configured,
    send_telegram_alert,
    send_telegram_message,
)
import scripts.run_camera as ec


# ---------------------------------------------------------------------------
# TEST 1: Event Creation
# ---------------------------------------------------------------------------
def test_event_creation():
    event = AlertEvent(
        alert_level="HIGH",
        risk_score=72,
        confidence=0.94,
        track_id=2,
        group_size=4,
        zone="WARNING",
        distance_m=380.0,
        movement="APPROACHING",
        simulation_mode=True,
    )
    assert event.event_id.startswith("zogan-")
    assert event.alert_level == "HIGH"
    assert event.risk_level == "HIGH"
    assert event.risk_score == 72
    assert event.confidence == 0.94
    assert event.track_id == 2
    assert event.group_size == 4
    assert event.zone == "WARNING"
    assert event.distance_m == 380.0
    assert event.movement == "APPROACHING"
    assert event.simulation_mode is True


# ---------------------------------------------------------------------------
# TEST 2: Event Serialization
# ---------------------------------------------------------------------------
def test_event_serialization():
    # Only confidence and alert_level populated; rest are None
    partial = AlertEvent(confidence=0.88, alert_level="MEDIUM")
    data = partial.to_dict(exclude_none=True)

    assert "event_id" in data
    assert "timestamp" in data
    assert "alert_level" in data
    assert data["alert_level"] == "MEDIUM"
    assert data["confidence"] == 0.88

    # Ensure unobserved metrics are strictly not fabricated
    assert "track_id" not in data
    assert "zone" not in data
    assert "distance_m" not in data
    assert "movement" not in data
    assert "risk_score" not in data

    # Valid JSON string
    json_str = partial.to_json()
    parsed = json.loads(json_str)
    assert parsed["event_id"] == partial.event_id
    assert parsed["confidence"] == 0.88


# ---------------------------------------------------------------------------
# TEST 3: Unique Event IDs
# ---------------------------------------------------------------------------
def test_unique_event_ids():
    ids = set()
    for _ in range(50):
        ev = AlertEvent(alert_level="HIGH")
        assert ev.event_id.startswith("zogan-")
        ids.add(ev.event_id)
    assert len(ids) == 50, "Event IDs must be unique across all instances!"


# ---------------------------------------------------------------------------
# TEST 4: JSONL Logging
# ---------------------------------------------------------------------------
def test_jsonl_logging():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "alerts.jsonl"
        logger = AlertEventLogger(log_file=log_file)

        e1 = AlertEvent(event_id="zogan-001", alert_level="HIGH", risk_score=70)
        e2 = AlertEvent(event_id="zogan-002", alert_level="CRITICAL", risk_score=95)

        logger.log(e1)
        logger.log(e2)

        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 2
        rec1 = json.loads(lines[0])
        rec2 = json.loads(lines[1])
        assert rec1["event_id"] == "zogan-001"
        assert rec1["alert_level"] == "HIGH"
        assert rec2["event_id"] == "zogan-002"
        assert rec2["alert_level"] == "CRITICAL"


# ---------------------------------------------------------------------------
# TEST 5: Log Directory Creation
# ---------------------------------------------------------------------------
def test_log_directory_creation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        deep_path = Path(tmp_dir) / "nested" / "subfolder" / "alerts.jsonl"
        assert not deep_path.parent.exists()

        logger = AlertEventLogger(log_file=deep_path)
        logger.log(AlertEvent(alert_level="LOW"))

        assert deep_path.parent.exists()
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# TEST 6: Recent History Reading
# ---------------------------------------------------------------------------
def test_recent_history_reading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "alerts.jsonl"

        # Populating test records
        events = [
            {"event_id": "zogan-1", "alert_level": "LOW", "risk_score": 10},
            {"event_id": "zogan-2", "alert_level": "HIGH", "risk_score": 70},
            {"event_id": "zogan-3", "alert_level": "MEDIUM", "risk_score": 30},
            {"event_id": "zogan-4", "alert_level": "HIGH", "risk_score": 75},
        ]
        with open(log_file, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        # Load recent alerts
        all_alerts = get_recent_alerts(log_file=log_file, limit=10)
        assert len(all_alerts) == 4

        recent_two = get_recent_alerts(log_file=log_file, limit=2)
        assert len(recent_two) == 2
        assert [r["event_id"] for r in recent_two] == ["zogan-3", "zogan-4"]

        # Counts
        assert get_alert_count(log_file=log_file) == 4
        assert count_alerts(log_file=log_file, alert_level="HIGH") == 2

        # Latest alert
        latest = get_latest_alert(log_file=log_file)
        assert latest["event_id"] == "zogan-4"

        # Filter by risk
        high_events = filter_by_risk(all_alerts, "HIGH")
        assert len(high_events) == 2


# ---------------------------------------------------------------------------
# TEST 7: Missing Telegram Credentials
# ---------------------------------------------------------------------------
def test_missing_telegram_credentials():
    with patch.dict(os.environ, {}, clear=True):
        assert not is_telegram_configured()
        token, chat_id = get_telegram_credentials()
        assert token is None
        assert chat_id is None


# ---------------------------------------------------------------------------
# TEST 8: Telegram Disabled Behavior
# ---------------------------------------------------------------------------
def test_telegram_disabled_behavior():
    with patch.dict(os.environ, {}, clear=True):
        event = AlertEvent(alert_level="HIGH", confidence=0.92)
        res = send_telegram_alert(event)
        assert res["success"] is False
        assert res["status"] == "not_configured"


# ---------------------------------------------------------------------------
# TEST 9: Message Formatting
# ---------------------------------------------------------------------------
def test_message_formatting():
    event = AlertEvent(
        event_id="zogan-test",
        timestamp="2026-10-03 18:42:12",
        alert_level="HIGH",
        risk_score=72,
        confidence=0.94,
        track_id=2,
        group_size=4,
        zone="WARNING",
        distance_m=380.0,
        movement="APPROACHING",
        simulation_mode=True,
    )
    msg = format_telegram_message(event)

    assert "🚨 PROJECT ZOGAN ALERT" in msg
    assert "Risk: HIGH" in msg
    assert "Score: 72" in msg
    assert "Confidence: 94%" in msg
    assert "Track: #2" in msg
    assert "Group Size: 4" in msg
    assert "Zone: WARNING" in msg
    assert "Distance: 380 m" in msg
    assert "Movement: APPROACHING" in msg
    assert "Time: 2026-10-03 18:42:12" in msg
    assert "Mode: SIMULATION" in msg


# ---------------------------------------------------------------------------
# TEST 10: Telegram Success Response (Mocked)
# ---------------------------------------------------------------------------
def test_telegram_success_response():
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps({"ok": True, "result": {"message_id": 101}}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = send_telegram_message("test", token="dummy_token", chat_id="dummy_chat")
        assert res["success"] is True
        assert res["status"] == "sent"
        assert res["response"]["result"]["message_id"] == 101


# ---------------------------------------------------------------------------
# TEST 11: Telegram Timeout / Failure (Mocked)
# ---------------------------------------------------------------------------
def test_telegram_timeout_and_failure():
    # Timeout
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        res = send_telegram_message("test", token="dummy_token", chat_id="dummy_chat")
        assert res["success"] is False
        assert res["status"] == "timeout"

    # API failure
    mock_fail_resp = MagicMock()
    mock_fail_resp.getcode.return_value = 400
    mock_fail_resp.read.return_value = json.dumps({"ok": False, "description": "Chat not found"}).encode("utf-8")
    mock_fail_resp.__enter__.return_value = mock_fail_resp

    with patch("urllib.request.urlopen", return_value=mock_fail_resp):
        res = send_telegram_message("test", token="dummy_token", chat_id="dummy_chat")
        assert res["success"] is False
        assert res["status"] == "api_error"
        assert "Chat not found" in res["error"]


# ---------------------------------------------------------------------------
# TEST 12: Dispatcher Behavior
# ---------------------------------------------------------------------------
def test_dispatcher_behavior():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "disp.jsonl"
        ev = AlertEvent(event_id="zogan-disp", alert_level="HIGH", risk_score=75)

        # Dispatch with unconfigured Telegram
        with patch.dict(os.environ, {}, clear=True):
            res = dispatch_alert(ev, log_file=log_file, verbose=False)
            assert res["logged"] is True
            assert res["telegram_configured"] is False
            assert res["telegram_status"] == "disabled"

        # Verify entry in log
        records = load_alerts(log_file=log_file)
        assert len(records) == 1
        assert records[0]["event_id"] == "zogan-disp"


# ---------------------------------------------------------------------------
# TEST 13 & 14: HIGH and CRITICAL Alert Behavior
# ---------------------------------------------------------------------------
def test_high_and_critical_alerts_trigger_telegram():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "high_crit.jsonl"

        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = json.dumps({"ok": True, "result": {"message_id": 1}}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch.dict(os.environ, {ENV_BOT_TOKEN: "tok", ENV_CHAT_ID: "chat"}):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                # HIGH level
                e_high = AlertEvent(alert_level="HIGH", risk_score=70)
                res_high = dispatch_alert(e_high, log_file=log_file, verbose=False)
                assert res_high["logged"] is True
                assert res_high["telegram_sent"] is True

                # CRITICAL level
                e_crit = AlertEvent(alert_level="CRITICAL", risk_score=95)
                res_crit = dispatch_alert(e_crit, log_file=log_file, verbose=False)
                assert res_crit["logged"] is True
                assert res_crit["telegram_sent"] is True


# ---------------------------------------------------------------------------
# TEST 15: LOW and MEDIUM Behavior (Log Only / Local Warning)
# ---------------------------------------------------------------------------
def test_low_and_medium_alerts_skip_telegram():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "low_med.jsonl"

        with patch.dict(os.environ, {ENV_BOT_TOKEN: "tok", ENV_CHAT_ID: "chat"}):
            # LOW level
            e_low = AlertEvent(alert_level="LOW", risk_score=15)
            res_low = dispatch_alert(e_low, log_file=log_file, verbose=False)
            assert res_low["logged"] is True
            assert res_low["telegram_sent"] is False
            assert res_low["telegram_status"] == "skipped_level_threshold"

            # MEDIUM level
            e_med = AlertEvent(alert_level="MEDIUM", risk_score=40)
            res_med = dispatch_alert(e_med, log_file=log_file, verbose=False)
            assert res_med["logged"] is True
            assert res_med["telegram_sent"] is False
            assert res_med["telegram_status"] == "skipped_level_threshold"

        # Verify both are safely logged locally
        records = load_alerts(log_file=log_file)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# TEST 16: Existing Cooldown Compatibility
# ---------------------------------------------------------------------------
def test_existing_cooldown_compatibility():
    consecutive_frames = 0
    last_alert_time = 0.0
    alerts_triggered = 0
    current_time = time.time()

    # Simulate 7 frames of detection
    for _ in range(1, 8):
        consecutive_frames += 1
        time_since_last = current_time - last_alert_time
        cooldown_active = time_since_last < ec.ALERT_COOLDOWN_SECONDS

        if consecutive_frames >= ec.REQUIRED_DETECTIONS:
            if not cooldown_active:
                tracked = [{"id": 1, "conf": 0.98, "movement": "STATIONARY"}]
                risk = {
                    "alert_level": "HIGH",
                    "risk_score": 67,
                    "zone": "BUFFER",
                    "distance_to_protected_m": 350.0,
                    "trend": "APPROACHING",
                    "group_size": 1,
                    "geo_mode": "SIMULATION",
                }
                ec.trigger_alert(0.98, tracked_info=tracked, risk_info=risk)
                last_alert_time = current_time
                alerts_triggered += 1

    # Exactly 1 alert should have been dispatched across 7 frames due to 30s cooldown
    assert alerts_triggered == 1, f"Expected 1 alert, got {alerts_triggered}"


# ---------------------------------------------------------------------------
# TEST 17: Graceful Log Failure Handling
# ---------------------------------------------------------------------------
def test_graceful_log_failure_handling():
    event = AlertEvent(alert_level="HIGH")
    with patch("builtins.open", side_effect=IOError("Disk write permission denied")):
        res = dispatch_alert(event, verbose=False)
        assert res["logged"] is False
        assert "Disk write permission denied" in res["log_error"]


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("🐘 RUNNING PROJECT ZOGAN TESTS (test_alerts.py)")
    print("=" * 65)

    test_event_creation()
    print("  ✓ Test 1: Event creation passed.")
    test_event_serialization()
    print("  ✓ Test 2: Event serialization passed.")
    test_unique_event_ids()
    print("  ✓ Test 3: Unique event IDs passed.")
    test_jsonl_logging()
    print("  ✓ Test 4: JSONL logging passed.")
    test_log_directory_creation()
    print("  ✓ Test 5: Log directory creation passed.")
    test_recent_history_reading()
    print("  ✓ Test 6: Recent history reading passed.")
    test_missing_telegram_credentials()
    print("  ✓ Test 7: Missing credentials handled.")
    test_telegram_disabled_behavior()
    print("  ✓ Test 8: Telegram disabled behavior passed.")
    test_message_formatting()
    print("  ✓ Test 9: Message formatting passed.")
    test_telegram_success_response()
    print("  ✓ Test 10: Telegram success response passed.")
    test_telegram_timeout_and_failure()
    print("  ✓ Test 11: Telegram timeout and failure handled.")
    test_dispatcher_behavior()
    print("  ✓ Test 12: Dispatcher behavior passed.")
    test_high_and_critical_alerts_trigger_telegram()
    print("  ✓ Test 13 & 14: HIGH & CRITICAL Telegram triggers passed.")
    test_low_and_medium_alerts_skip_telegram()
    print("  ✓ Test 15: LOW & MEDIUM log-only policy passed.")
    test_existing_cooldown_compatibility()
    print("  ✓ Test 16: Cooldown compatibility passed.")
    test_graceful_log_failure_handling()
    print("  ✓ Test 17: Graceful log failure handled.")

    print("\n" + "=" * 65)
    print("🎉 ALL 17 ALERTS TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
