"""
=============================================================================
🐘 PROJECT ZOGAN — AUTOMATED TEST SUITE FOR PHASE 6
=============================================================================

Comprehensive test verification for Event Logging & Remote Alert System:
  1. AlertEvent model creation, serialization, and JSON generation
  2. Non-existent values are omitted (no fabricated values)
  3. Deserialization from dictionary (AlertEvent.from_dict)
  4. create_alert_event factory function (backward and forward compatibility)
  5. Safe log directory creation (logs/ created if missing, never crashes)
  6. Local JSON Lines logging (logs/alerts.jsonl)
  7. Multi-event append integrity (valid individual JSON Lines)
  8. Corrupted line tolerance in history loader
  9. History reader: load, count, get_latest, filter_by_risk, and limit
 10. History reader behavior when log file does not exist (graceful return)
 11. Telegram message formatting with complete event metrics
 12. Telegram message formatting with partial fields (only present fields)
 13. Telegram credential detection and validation from environment
 14. Telegram service gracefully skips when unconfigured (no crash)
 15. Telegram network mock delivery & HTTP error recovery (fail-safe)
 16. Unified Dispatcher workflow (local log + remote dispatch coordination)
 17. Pipeline integration: elephant_camera.trigger_alert backward compatibility

Usage:
    python test_phase6.py
=============================================================================
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Project imports
from alerts.dispatcher import dispatch_alert
from alerts.event_logger import AlertEventLogger
from alerts.history import count_alerts, filter_by_risk, get_latest_alert, load_alerts
from alerts.models import AlertEvent, create_alert_event
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


def test_alert_event_model():
    print("\n--- TEST 1: AlertEvent Model & Serialization ---")
    event = AlertEvent(
        event_id="evt_test_12345",
        timestamp="2026-10-03T12:00:00Z",
        risk_level="HIGH",
        risk_score=72,
        confidence=0.9412,
        track_id=2,
        group_size=4,
        zone="WARNING",
        distance_m=380.0,
        movement="APPROACHING",
        simulation=True,
    )

    data = event.to_dict()
    assert data["event_id"] == "evt_test_12345"
    assert data["risk_level"] == "HIGH"
    assert data["risk_score"] == 72
    assert data["confidence"] == 0.9412
    assert data["track_id"] == 2
    assert data["group_size"] == 4
    assert data["zone"] == "WARNING"
    assert data["distance_m"] == 380.0
    assert data["movement"] == "APPROACHING"
    assert data["simulation"] is True

    json_str = event.to_json()
    parsed = json.loads(json_str)
    assert parsed["event_id"] == "evt_test_12345"
    assert parsed["risk_score"] == 72
    print("  ✓ Full AlertEvent correctly instantiated, serialized to dict, and exported to JSON.")


def test_no_fabricated_values():
    print("\n--- TEST 2: Missing Data is Omitted (No Fabricated Values) ---")
    # Partial event without track, zone, distance, or movement
    partial_event = AlertEvent(
        event_id="evt_partial_01",
        confidence=0.88,
        risk_level="MEDIUM",
    )
    data = partial_event.to_dict(exclude_none=True)

    assert "event_id" in data
    assert "confidence" in data
    assert "risk_level" in data
    assert "track_id" not in data, "Non-existent track_id must not be fabricated!"
    assert "zone" not in data, "Non-existent zone must not be fabricated!"
    assert "distance_m" not in data, "Non-existent distance_m must not be fabricated!"
    assert "movement" not in data, "Non-existent movement must not be fabricated!"
    assert "simulation" not in data, "Non-existent simulation must not be fabricated!"
    print("  ✓ Verified: Keys with None values are excluded; no fabricated metrics are emitted.")


def test_deserialization_from_dict():
    print("\n--- TEST 3: Deserialization from Dict (AlertEvent.from_dict) ---")
    payload = {
        "event_id": "evt_from_dict_99",
        "timestamp": "2026-10-03T14:30:00Z",
        "risk_level": "CRITICAL",
        "risk_score": 85,
        "confidence": 0.96,
        "track_id": 5,
        "group_size": 2,
        "zone": "VILLAGE",
        "distance_m": 0.0,
        "movement": "APPROACHING",
        "simulation": False,
    }
    event = AlertEvent.from_dict(payload)
    assert event.event_id == "evt_from_dict_99"
    assert event.risk_level == "CRITICAL"
    assert event.risk_score == 85
    assert event.distance_m == 0.0
    assert event.simulation is False
    print("  ✓ Successfully reconstructed AlertEvent object from serialized dict.")


def test_create_alert_event_factory():
    print("\n--- TEST 4: create_alert_event Factory Function ---")
    # Scenario A: Phase 2 minimal style (confidence only)
    e1 = create_alert_event(confidence=0.92)
    assert e1.confidence == 0.92
    assert e1.risk_level == "HIGH"  # Default level when no risk engine is attached
    assert e1.track_id is None

    # Scenario B: Phase 5 enriched style
    tracked = [{"id": 7, "conf": 0.95, "movement": "RIGHT"}]
    risk = {
        "risk_level": "HIGH",
        "risk_score": 68,
        "zone": "BUFFER",
        "distance_to_protected_m": 420.0,
        "trend": "APPROACHING",
        "group_size": 1,
        "geo_mode": "SIMULATION",
    }
    e2 = create_alert_event(confidence=0.95, tracked_info=tracked, risk_info=risk)
    assert e2.track_id == 7
    assert e2.risk_score == 68
    assert e2.zone == "BUFFER"
    assert e2.distance_m == 420.0
    assert e2.movement == "APPROACHING"  # Geographic trend preferred over image-space
    assert e2.simulation is True
    print("  ✓ Factory handles minimal Phase 2 inputs as well as fully-enriched Phase 5 inputs.")


def test_safe_log_directory_creation():
    print("\n--- TEST 5: Safe Log Directory Creation ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        deep_log_file = Path(tmp_dir) / "nested_dir" / "deeper" / "alerts.jsonl"
        assert not deep_log_file.parent.exists()

        logger = AlertEventLogger(log_file=deep_log_file)
        event = AlertEvent(event_id="evt_dir_test", risk_level="LOW", confidence=0.75)
        logger.log(event)

        assert deep_log_file.parent.exists(), "Logger failed to create missing parent directory!"
        assert deep_log_file.exists(), "Logger failed to create log file!"
        print(f"  ✓ Safely created non-existent directory tree: {deep_log_file.parent}")


def test_jsonl_logging_and_append():
    print("\n--- TEST 6 & 7: Local JSONL Logging & Append Mode ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "alerts.jsonl"
        logger = AlertEventLogger(log_file=log_path)

        e1 = AlertEvent(event_id="evt_1", risk_level="MEDIUM", risk_score=40, confidence=0.82)
        e2 = AlertEvent(event_id="evt_2", risk_level="HIGH", risk_score=70, confidence=0.95)

        logger.log(e1)
        logger.log(e2)

        with open(log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        record1 = json.loads(lines[0])
        record2 = json.loads(lines[1])

        assert record1["event_id"] == "evt_1"
        assert record1["risk_level"] == "MEDIUM"
        assert record2["event_id"] == "evt_2"
        assert record2["risk_level"] == "HIGH"
        print(f"  ✓ Logged {len(lines)} independent events atomically in valid JSON Lines format.")


def test_corrupted_lines_tolerance():
    print("\n--- TEST 8: Corrupted / Partial Line Tolerance ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "alerts.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write('{"event_id": "valid_1", "risk_level": "LOW"}\n')
            f.write("CORRUPTED_JSON_LINE_ERROR_GARBAGE\n")
            f.write('{"event_id": "valid_2", "risk_level": "CRITICAL"}\n')
            f.write("\n")  # Empty line

        records = load_alerts(log_file=log_path)
        assert len(records) == 2, f"Expected 2 valid records, got {len(records)}"
        assert records[0]["event_id"] == "valid_1"
        assert records[1]["event_id"] == "valid_2"
        print("  ✓ Corrupted lines and empty lines gracefully skipped without raising exceptions.")


def test_history_queries():
    print("\n--- TEST 9 & 10: Alert History Query Utilities ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "alerts.jsonl"

        # Test on non-existent file
        assert load_alerts(log_file=log_path) == []
        assert count_alerts(log_file=log_path) == 0
        assert get_latest_alert(log_file=log_path) is None

        # Populate with test events
        events = [
            {"event_id": "e1", "risk_level": "LOW", "risk_score": 15},
            {"event_id": "e2", "risk_level": "HIGH", "risk_score": 65},
            {"event_id": "e3", "risk_level": "MEDIUM", "risk_score": 35},
            {"event_id": "e4", "risk_level": "HIGH", "risk_score": 74},
            {"event_id": "e5", "risk_level": "CRITICAL", "risk_score": 90},
        ]
        with open(log_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        # Total count
        assert count_alerts(log_file=log_path) == 5
        # Filter by risk level
        high_alerts = load_alerts(log_file=log_path, risk_level="HIGH")
        assert len(high_alerts) == 2
        assert count_alerts(log_file=log_path, risk_level="HIGH") == 2
        # Latest alert
        latest = get_latest_alert(log_file=log_path)
        assert latest is not None
        assert latest["event_id"] == "e5"
        # Latest HIGH alert
        latest_high = get_latest_alert(log_file=log_path, risk_level="HIGH")
        assert latest_high is not None
        assert latest_high["event_id"] == "e4"
        # Limit to 2 most recent
        recent_two = load_alerts(log_file=log_path, limit=2)
        assert len(recent_two) == 2
        assert [r["event_id"] for r in recent_two] == ["e4", "e5"]
        # filter_by_risk helper
        filtered = filter_by_risk(events, "CRITICAL")
        assert len(filtered) == 1 and filtered[0]["event_id"] == "e5"

        print("  ✓ load_alerts, count_alerts, get_latest_alert, and filter_by_risk all behave correctly.")


def test_telegram_message_formatting_full():
    print("\n--- TEST 11: Telegram Message Formatting (Complete Event) ---")
    event = AlertEvent(
        event_id="evt_telegram_demo",
        timestamp="2026-10-03 14:45:00",
        risk_level="HIGH",
        risk_score=72,
        confidence=0.94,
        track_id=2,
        group_size=4,
        zone="WARNING",
        distance_m=380.0,
        movement="APPROACHING",
        simulation=True,
    )
    msg = format_telegram_message(event)

    expected_fragments = [
        "🚨 PROJECT ZOGAN ALERT",
        "Risk: HIGH",
        "Score: 72",
        "Confidence: 94%",
        "Track: #2",
        "Group Size: 4",
        "Zone: WARNING",
        "Distance: 380 m",
        "Movement: APPROACHING",
        "Time: 2026-10-03 14:45:00",
        "Mode: SIMULATION",
    ]

    for frag in expected_fragments:
        assert frag in msg, f"Missing fragment in formatted Telegram message: '{frag}'"

    print("  ✓ Complete event formatted into Telegram message according to specification:\n")
    for line in msg.split("\n"):
        print(f"    | {line}")


def test_telegram_message_formatting_partial():
    print("\n--- TEST 12: Telegram Message Formatting (Partial Event) ---")
    # Event with ONLY confidence, risk_level, and time
    partial_event = AlertEvent(
        confidence=0.91,
        risk_level="CRITICAL",
        timestamp="2026-10-03 15:00:00",
    )
    msg = format_telegram_message(partial_event)

    assert "🚨 PROJECT ZOGAN ALERT" in msg
    assert "Risk: CRITICAL" in msg
    assert "Confidence: 91%" in msg
    assert "Time: 2026-10-03 15:00:00" in msg

    # Verify missing fields are strictly omitted
    assert "Score:" not in msg
    assert "Track:" not in msg
    assert "Group Size:" not in msg
    assert "Zone:" not in msg
    assert "Distance:" not in msg
    assert "Movement:" not in msg
    assert "Mode:" not in msg
    print("  ✓ Partial event formats only available values without placeholder or fabricated lines.")


def test_telegram_credentials_detection():
    print("\n--- TEST 13: Telegram Credential Detection from Environment ---")
    with patch.dict(os.environ, {}, clear=True):
        assert not is_telegram_configured()
        token, chat_id = get_telegram_credentials()
        assert token is None and chat_id is None

    with patch.dict(
        os.environ, {ENV_BOT_TOKEN: "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", ENV_CHAT_ID: "-100123456789"}
    ):
        assert is_telegram_configured()
        token, chat_id = get_telegram_credentials()
        assert token == "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        assert chat_id == "-100123456789"

    # Template placeholder filtering
    with patch.dict(os.environ, {ENV_BOT_TOKEN: "YOUR_BOT_TOKEN", ENV_CHAT_ID: ""}):
        assert not is_telegram_configured()
    print("  ✓ Environment variables correctly detected and placeholder tokens sanitized.")


def test_telegram_unconfigured_safety():
    print("\n--- TEST 14: Telegram Fail-Safe When Unconfigured ---")
    with patch.dict(os.environ, {}, clear=True):
        event = AlertEvent(risk_level="HIGH", confidence=0.92)
        res = send_telegram_alert(event)
        assert res["success"] is False
        assert res["status"] == "not_configured"
        print("  ✓ send_telegram_alert returns safe error dictionary without crashing when unconfigured.")


def test_telegram_network_mock():
    print("\n--- TEST 15: Telegram Mock Delivery & Network Error Recovery ---")
    # Subtest A: Mock successful Telegram API delivery
    mock_success_response = MagicMock()
    mock_success_response.getcode.return_value = 200
    mock_success_response.read.return_value = json.dumps({"ok": True, "result": {"message_id": 42}}).encode("utf-8")
    mock_success_response.__enter__.return_value = mock_success_response

    with patch("urllib.request.urlopen", return_value=mock_success_response):
        res = send_telegram_message("test message", token="fake_token", chat_id="fake_chat_id")
        assert res["success"] is True
        assert res["status"] == "sent"
        assert res["response"]["result"]["message_id"] == 42
        print("  ✓ Mocked HTTP 200 Telegram delivery successfully processed.")

    # Subtest B: Mock network exception / timeout
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        res = send_telegram_message("test message", token="fake_token", chat_id="fake_chat_id")
        assert res["success"] is False
        assert res["status"] == "timeout"
        print("  ✓ Network timeout gracefully handled without unhandled exception.")


def test_unified_dispatcher():
    print("\n--- TEST 16: Unified Alert Dispatcher Workflow ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "dispatcher_alerts.jsonl"
        event = AlertEvent(
            event_id="evt_disp_01",
            risk_level="HIGH",
            risk_score=75,
            confidence=0.97,
            track_id=1,
            zone="BUFFER",
        )

        # Dispatch with Telegram unconfigured
        with patch.dict(os.environ, {}, clear=True):
            dispatch_result = dispatch_alert(event, log_file=log_path)
            assert dispatch_result["logged"] is True
            assert dispatch_result["telegram_configured"] is False
            assert dispatch_result["telegram_status"] in ("disabled", "skipped_not_configured")

        # Verify event was written to JSONL
        records = load_alerts(log_file=log_path)
        assert len(records) == 1
        assert records[0]["event_id"] == "evt_disp_01"
        assert records[0]["risk_score"] == 75

        # Dispatch with mock Telegram configured
        mock_success_response = MagicMock()
        mock_success_response.getcode.return_value = 200
        mock_success_response.read.return_value = json.dumps({"ok": True, "result": {"message_id": 99}}).encode("utf-8")
        mock_success_response.__enter__.return_value = mock_success_response

        with patch.dict(os.environ, {ENV_BOT_TOKEN: "mock_tok", ENV_CHAT_ID: "mock_chat"}):
            with patch("urllib.request.urlopen", return_value=mock_success_response):
                e2 = AlertEvent(event_id="evt_disp_02", risk_level="CRITICAL", risk_score=95)
                res2 = dispatch_alert(e2, log_file=log_path)
                assert res2["logged"] is True
                assert res2["telegram_configured"] is True
                assert res2["telegram_sent"] is True
                assert res2["telegram_status"] == "sent"

        records_updated = load_alerts(log_file=log_path)
        assert len(records_updated) == 2
        print("  ✓ Unified dispatcher recorded both events to JSONL and coordinated Telegram delivery.")


def test_pipeline_integration():
    print("\n--- TEST 17: elephant_camera.trigger_alert Integration & Backward Compatibility ---")
    # Verify Phase 2 call signature (single confidence argument)
    res_phase2 = ec.trigger_alert(0.93)
    assert res_phase2 is not None
    assert res_phase2["logged"] is True
    assert res_phase2["event_id"] is not None

    # Verify Phase 5 call signature (confidence + tracked_info + risk_info)
    tracked = [{"id": 3, "conf": 0.96, "movement": "APPROACHING"}]
    risk = {
        "risk_level": "HIGH",
        "risk_score": 71,
        "zone": "WARNING",
        "distance_to_protected_m": 390.0,
        "trend": "APPROACHING",
        "group_size": 2,
        "geo_mode": "SIMULATION",
    }
    res_phase5 = ec.trigger_alert(0.96, tracked_info=tracked, risk_info=risk)
    assert res_phase5 is not None
    assert res_phase5["logged"] is True

    # Check that both alerts were logged to logs/alerts.jsonl
    latest = get_latest_alert()
    assert latest is not None
    assert latest["risk_score"] == 71
    assert latest["track_id"] == 3
    print("  ✓ elephant_camera.trigger_alert backward compatibility and end-to-end logging verified.")


def run_all_tests():
    print("=" * 65)
    print("🐘 RUNNING PROJECT ZOGAN PHASE 6 ALERT & LOGGING TEST SUITE")
    print("=" * 65)

    test_alert_event_model()
    test_no_fabricated_values()
    test_deserialization_from_dict()
    test_create_alert_event_factory()
    test_safe_log_directory_creation()
    test_jsonl_logging_and_append()
    test_corrupted_lines_tolerance()
    test_history_queries()
    test_telegram_message_formatting_full()
    test_telegram_message_formatting_partial()
    test_telegram_credentials_detection()
    test_telegram_unconfigured_safety()
    test_telegram_network_mock()
    test_unified_dispatcher()
    test_pipeline_integration()

    print("\n" + "=" * 65)
    print("🎉 ALL 17 PHASE 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
