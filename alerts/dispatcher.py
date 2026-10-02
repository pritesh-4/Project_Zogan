"""
=============================================================================
🐘 PROJECT ZOGAN — UNIFIED ALERT DISPATCHER (PHASE 6)
=============================================================================

This module provides the single unified entry point for confirmed alert events:
  dispatch_alert(event)

Architectural Flow:
  Detection
      ↓
  Risk engine
      ↓
  Alert event
      ↓
  Local log (logs/alerts.jsonl)
      ↓
  Telegram (if configured & alert level >= threshold)

Alert-Level Policy:
  LOW:      log only
  MEDIUM:   log only (local warning)
  HIGH:     log + local alert + Telegram (if configured)
  CRITICAL: log + local alert + Telegram (if configured)

Pipeline Reliability Requirements:
  1. Record the incident locally in logs/alerts.jsonl.
  2. Attempt remote notification via Telegram if credentials are set and
     the event satisfies the configured alert-level threshold.
  3. NEVER crash the camera/detection pipeline if logging or Telegram fails.
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from alerts.event_logger import log_alert
from alerts.models import AlertEvent
from alerts.telegram import is_telegram_configured, send_telegram_alert

# Numeric priority mapping for alert levels
LEVEL_PRIORITY: Dict[str, int] = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

DEFAULT_MIN_TELEGRAM_LEVEL: str = "HIGH"


def dispatch_alert(
    event: Union[AlertEvent, Dict[str, Any]],
    log_file: Optional[Union[str, Path]] = None,
    send_telegram: bool = True,
    min_telegram_level: str = DEFAULT_MIN_TELEGRAM_LEVEL,
    timeout: float = 10.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Unified entry point to process and dispatch a confirmed alert event.

    Steps:
      1. Normalize the event into an AlertEvent object.
      2. Record event locally to JSONL file (creates directory safely).
      3. Evaluate alert-level policy (default: HIGH & CRITICAL trigger Telegram).
      4. Attempt remote notification via Telegram if configured and allowed.
      5. Return a structured dispatch result summary.

    Guaranteed fail-safe; never crashes the caller or detection loop.
    """
    # 1. Normalize input event
    alert_obj: AlertEvent
    if isinstance(event, AlertEvent):
        alert_obj = event
    elif isinstance(event, dict):
        alert_obj = AlertEvent.from_dict(event)
    else:
        alert_obj = AlertEvent()

    event_level = (alert_obj.alert_level or alert_obj.risk_level or "HIGH").upper()
    event_priority = LEVEL_PRIORITY.get(event_level, 3)
    min_priority = LEVEL_PRIORITY.get(min_telegram_level.upper(), 3)

    result: Dict[str, Any] = {
        "event_id": alert_obj.event_id,
        "alert_level": event_level,
        "risk_level": event_level,
        "logged": False,
        "log_error": None,
        "telegram_configured": False,
        "telegram_attempted": False,
        "telegram_sent": False,
        "telegram_status": "not_attempted",
        "telegram_error": None,
    }

    # 2. Local Event Logging (logs/alerts.jsonl)
    try:
        logged_record = log_alert(alert_obj, log_file=log_file)
        result["logged"] = True
        result["logged_record"] = logged_record
    except Exception as e:
        result["log_error"] = str(e)
        if verbose:
            print(f"[ALERT LOGGER ERROR] Failed to record alert locally: {e}")

    # 3. Remote Telegram Dispatch Policy
    if not send_telegram:
        result["telegram_status"] = "disabled"
    elif event_priority < min_priority:
        # Policy: LOW/MEDIUM are logged locally without remote notification
        result["telegram_status"] = "skipped_level_threshold"
    else:
        try:
            telegram_ready = is_telegram_configured()
            result["telegram_configured"] = telegram_ready

            if telegram_ready:
                result["telegram_attempted"] = True
                tele_res = send_telegram_alert(alert_obj, timeout=timeout)
                result["telegram_sent"] = tele_res.get("success", False)
                result["telegram_status"] = tele_res.get("status", "unknown")
                if not tele_res.get("success"):
                    result["telegram_error"] = tele_res.get("error")
            else:
                result["telegram_status"] = "disabled"
        except Exception as e:
            result["telegram_status"] = "failed"
            result["telegram_error"] = str(e)

    # 4. Observability Terminal Output
    if verbose:
        status_tag = result["telegram_status"].upper()
        if result["telegram_sent"]:
            tele_display = "SENT"
        elif result["telegram_status"] == "disabled":
            tele_display = "DISABLED"
        elif result["telegram_status"] == "skipped_level_threshold":
            tele_display = f"SKIPPED (Level {event_level} < {min_telegram_level})"
        else:
            err_info = f" — {result.get('telegram_error')}" if result.get("telegram_error") else ""
            tele_display = f"FAILED ({status_tag}{err_info})"

        print(f"[ALERT DISPATCHED] Event ID: {alert_obj.event_id} | Level: {event_level} | Telegram: {tele_display}")

    return result
