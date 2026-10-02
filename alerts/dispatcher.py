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
  Telegram (if configured)

Pipeline Reliability Requirements:
  1. Record the incident locally in logs/alerts.jsonl.
  2. Attempt remote notification via Telegram if credentials are set.
  3. NEVER crash the camera/detection pipeline if logging or Telegram fails.
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from alerts.event_logger import log_alert
from alerts.models import AlertEvent
from alerts.telegram import is_telegram_configured, send_telegram_alert


def dispatch_alert(
    event: Union[AlertEvent, Dict[str, Any]],
    log_file: Optional[Union[str, Path]] = None,
    send_telegram: bool = True,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Unified entry point to process and dispatch a confirmed alert event.

    Steps:
      1. Normalize the event into an AlertEvent object.
      2. Record event locally to JSONL file (creates directory safely).
      3. Attempt remote notification via Telegram if credentials are set.
      4. Return a structured dispatch result summary.

    This function is guaranteed to catch internal exceptions and never crash
    the caller or detection loop.
    """
    # 1. Normalize input event
    alert_obj: AlertEvent
    if isinstance(event, AlertEvent):
        alert_obj = event
    elif isinstance(event, dict):
        alert_obj = AlertEvent.from_dict(event)
    else:
        alert_obj = AlertEvent()

    result: Dict[str, Any] = {
        "event_id": alert_obj.event_id,
        "logged": False,
        "log_error": None,
        "telegram_configured": False,
        "telegram_attempted": False,
        "telegram_sent": False,
        "telegram_status": "not_attempted",
        "telegram_error": None,
    }

    # 2. Local Event Logging
    try:
        logged_record = log_alert(alert_obj, log_file=log_file)
        result["logged"] = True
        result["logged_record"] = logged_record
    except Exception as e:
        result["log_error"] = str(e)
        print(f"[ALERT LOGGER ERROR] Failed to record alert locally: {e}")

    # 3. Remote Telegram Dispatch
    if send_telegram:
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
                    print(f"[TELEGRAM WARNING] Remote alert failed: {tele_res.get('error')}")
                else:
                    print(f"[TELEGRAM] Remote alert successfully dispatched for Event {alert_obj.event_id}.")
            else:
                result["telegram_status"] = "skipped_not_configured"
        except Exception as e:
            result["telegram_status"] = "exception"
            result["telegram_error"] = str(e)
            print(f"[TELEGRAM EXCEPTION] Unexpected error during Telegram dispatch: {e}")
    else:
        result["telegram_status"] = "disabled"

    return result
