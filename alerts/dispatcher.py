"""
=============================================================================
🐘 PROJECT ZOGAN — UNIFIED ALERT DISPATCHER (PHASE 6 & PHASE 9)
=============================================================================

This module provides the single unified entry point for confirmed alert events:
  dispatch_alert(event)

Architectural Flow:
  Detection
      ↓
  Risk Engine
      ↓
  Alert Event
      ↓
  [1] LOCAL PERSISTENCE FIRST (logs/alerts.jsonl)
      ↓
  [2] REMOTE TELEGRAM NOTIFICATION (if configured & priority >= threshold)
      ↓
  [3] OFFLINE QUEUE (logs/delivery_queue.jsonl if delivery fails)
      ↓
  [4] RETRY WITH BOUNDED EXPONENTIAL BACKOFF

Reliability Guarantees:
  1. Local storage is strictly FIRST — remote network availability never
     dictates local incident persistence.
  2. Non-blocking delivery timeout ensures camera capture never stalls.
  3. Failed notifications are enqueued persistently for bounded retries.
  4. Stable event IDs guarantee deduplication across retries and restarts.
  5. Never crashes the video detection pipeline on any network/HTTP error.
=============================================================================
"""

import atexit
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import config
from alerts.event_logger import log_alert
from alerts.models import (
    DELIVERY_FAILED,
    DELIVERY_NOT_CONFIGURED,
    DELIVERY_PENDING,
    DELIVERY_SENT,
    DELIVERY_SKIPPED,
    AlertEvent,
)
from alerts.queue import PersistentAlertQueue
from alerts.telegram import is_telegram_configured, send_telegram_alert

# Numeric priority mapping for alert levels
LEVEL_PRIORITY: Dict[str, int] = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

DEFAULT_MIN_TELEGRAM_LEVEL: str = "HIGH"

# Global persistent queue singleton
_GLOBAL_QUEUE: Optional[PersistentAlertQueue] = None

# Thread pool for non-blocking asynchronous alert delivery
_DISPATCH_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_dispatch_executor() -> ThreadPoolExecutor:
    """Returns thread pool executor for background remote dispatching."""
    global _DISPATCH_EXECUTOR
    if _DISPATCH_EXECUTOR is None:
        _DISPATCH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="zogan_alert_worker")
    return _DISPATCH_EXECUTOR


def _shutdown_dispatch_executor() -> None:
    """Cleans up the dispatch executor upon process exit."""
    global _DISPATCH_EXECUTOR
    if _DISPATCH_EXECUTOR is not None:
        _DISPATCH_EXECUTOR.shutdown(wait=False)
        _DISPATCH_EXECUTOR = None


atexit.register(_shutdown_dispatch_executor)


def get_alert_queue() -> PersistentAlertQueue:
    """Returns the shared persistent alert delivery queue."""
    global _GLOBAL_QUEUE
    if _GLOBAL_QUEUE is None:
        _GLOBAL_QUEUE = PersistentAlertQueue()
    return _GLOBAL_QUEUE


def set_alert_queue(queue: Optional[PersistentAlertQueue]) -> None:
    """Sets or resets the shared persistent alert delivery queue (useful for testing)."""
    global _GLOBAL_QUEUE
    _GLOBAL_QUEUE = queue


def _execute_remote_attempt(
    alert_obj: AlertEvent,
    result: Dict[str, Any],
    active_queue: PersistentAlertQueue,
    effective_timeout: float,
    event_level: str,
    min_telegram_level: str,
    verbose: bool,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Helper to perform remote delivery attempt and offline queueing."""
    try:
        telegram_ready = is_telegram_configured()
        result["telegram_configured"] = telegram_ready

        if telegram_ready:
            result["telegram_attempted"] = True
            alert_obj.delivery_attempts = (alert_obj.delivery_attempts or 0) + 1
            alert_obj.last_delivery_attempt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            tele_res = send_telegram_alert(alert_obj, timeout=effective_timeout)
            result["telegram_sent"] = tele_res.get("success", False)
            result["telegram_status"] = tele_res.get("status", "unknown")

            if tele_res.get("success"):
                alert_obj.delivery_status = DELIVERY_SENT
                alert_obj.delivery_error = None
                result["delivery_status"] = DELIVERY_SENT
            else:
                # Remote delivery failed
                error_msg = tele_res.get("error") or "Unknown delivery error"
                alert_obj.delivery_status = DELIVERY_PENDING
                alert_obj.delivery_error = error_msg
                result["telegram_error"] = error_msg
                result["delivery_status"] = DELIVERY_PENDING

                # Enqueue into persistent offline queue for retry
                if getattr(config, "OFFLINE_QUEUE_ENABLED", True) and active_queue:
                    active_queue.enqueue(alert_obj, error=error_msg)
                    result["enqueued"] = True
                    result["pending_count"] = active_queue.pending_count
        else:
            alert_obj.delivery_status = DELIVERY_NOT_CONFIGURED
            result["telegram_status"] = "disabled"
            result["delivery_status"] = DELIVERY_NOT_CONFIGURED

    except Exception as e:
        alert_obj.delivery_status = DELIVERY_FAILED
        alert_obj.delivery_error = str(e)
        result["telegram_status"] = "failed"
        result["telegram_error"] = str(e)
        result["delivery_status"] = DELIVERY_FAILED
        if getattr(config, "OFFLINE_QUEUE_ENABLED", True) and active_queue:
            active_queue.enqueue(alert_obj, error=str(e))
            result["enqueued"] = True
            result["pending_count"] = active_queue.pending_count

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
            q_info = " [QUEUED FOR RETRY]" if result.get("enqueued") else ""
            tele_display = f"FAILED ({status_tag}{err_info}){q_info}"

        print(f"[ALERT DISPATCHED] Event ID: {alert_obj.event_id} | Level: {event_level} | Telegram: {tele_display}")

    if callback:
        try:
            callback(result)
        except Exception:
            pass

    return result


def dispatch_alert(
    event: Union[AlertEvent, Dict[str, Any]],
    log_file: Optional[Union[str, Path]] = None,
    send_telegram: bool = True,
    min_telegram_level: str = DEFAULT_MIN_TELEGRAM_LEVEL,
    timeout: Optional[float] = None,
    verbose: bool = True,
    queue: Optional[PersistentAlertQueue] = None,
    async_delivery: bool = False,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Unified entry point to process and dispatch a confirmed alert event.

    Order of Operations:
      1. Normalize the event into an AlertEvent object.
      2. Record event LOCALLY to JSONL file FIRST.
      3. Evaluate alert-level policy (default: HIGH & CRITICAL trigger Telegram).
      4. Attempt remote notification via Telegram with fast timeout (or async in background).
      5. If remote delivery fails, enqueue into persistent offline queue for retry.
      6. Return a structured dispatch result summary.

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

    # Use fast delivery timeout from config to avoid blocking video loop
    effective_timeout = timeout if timeout is not None else getattr(config, "ALERT_DELIVERY_TIMEOUT_SECONDS", 3.0)

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
        "delivery_status": DELIVERY_PENDING,
        "enqueued": False,
    }

    # 2. LOCAL EVENT LOGGING FIRST (logs/alerts.jsonl)
    # The fundamental principle: local event data must NEVER depend on network availability.
    try:
        logged_record = log_alert(alert_obj, log_file=log_file)
        result["logged"] = True
        result["logged_record"] = logged_record
    except Exception as e:
        result["log_error"] = str(e)
        if verbose:
            print(f"[ALERT LOGGER ERROR] Failed to record alert locally: {e}")

    # 3. Remote Telegram Dispatch Policy
    active_queue = queue if queue is not None else get_alert_queue()

    if not send_telegram:
        alert_obj.delivery_status = DELIVERY_NOT_CONFIGURED
        result["telegram_status"] = "disabled"
        result["delivery_status"] = DELIVERY_NOT_CONFIGURED
        if verbose:
            print(f"[ALERT DISPATCHED] Event ID: {alert_obj.event_id} | Level: {event_level} | Telegram: DISABLED")
        if callback:
            try:
                callback(result)
            except Exception:
                pass
        return result

    if event_priority < min_priority:
        # Policy: LOW/MEDIUM are logged locally without remote notification
        alert_obj.delivery_status = DELIVERY_SKIPPED
        result["telegram_status"] = "skipped_level_threshold"
        result["delivery_status"] = DELIVERY_SKIPPED
        if verbose:
            print(
                f"[ALERT DISPATCHED] Event ID: {alert_obj.event_id} | "
                f"Level: {event_level} | Telegram: SKIPPED (Level {event_level} < {min_telegram_level})"
            )
        if callback:
            try:
                callback(result)
            except Exception:
                pass
        return result

    # If async_delivery is requested, submit to background thread and return immediately
    if async_delivery:
        result["async_scheduled"] = True
        executor = _get_dispatch_executor()
        executor.submit(
            _execute_remote_attempt,
            alert_obj,
            result,
            active_queue,
            effective_timeout,
            event_level,
            min_telegram_level,
            verbose,
            callback,
        )
        return result

    # Otherwise execute synchronously
    return _execute_remote_attempt(
        alert_obj=alert_obj,
        result=result,
        active_queue=active_queue,
        effective_timeout=effective_timeout,
        event_level=event_level,
        min_telegram_level=min_telegram_level,
        verbose=verbose,
        callback=callback,
    )


def process_delivery_queue(
    queue: Optional[PersistentAlertQueue] = None,
    timeout: Optional[float] = None,
    now: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Processes and drains pending items in the offline queue whose retry delays
    have expired.

    Returns:
        Summary dict containing attempted, sent, failed, and remaining pending counts.
    """
    active_queue = queue if queue is not None else get_alert_queue()
    current_time = time.time() if now is None else now
    effective_timeout = timeout if timeout is not None else getattr(config, "ALERT_DELIVERY_TIMEOUT_SECONDS", 3.0)

    ready_items = active_queue.get_ready_for_retry(now=current_time)

    stats: Dict[str, Any] = {
        "ready_count": len(ready_items),
        "attempted": 0,
        "sent": 0,
        "failed": 0,
        "pending_remaining": active_queue.pending_count,
    }

    if not ready_items or not is_telegram_configured():
        return stats

    for item in ready_items:
        stats["attempted"] += 1
        try:
            tele_res = send_telegram_alert(item.event, timeout=effective_timeout)
            if tele_res.get("success"):
                active_queue.mark_sent(item.event.event_id, now=current_time)
                stats["sent"] += 1
                if verbose:
                    print(f"[RETRY SUCCESS] Delivered queued alert: {item.event.event_id}")
            else:
                err = tele_res.get("error", "Retry failed")
                active_queue.mark_attempt_failed(item.event.event_id, error=err, now=current_time)
                stats["failed"] += 1
                if verbose:
                    print(f"[RETRY FAILED] Alert {item.event.event_id}: {err}")
        except Exception as e:
            active_queue.mark_attempt_failed(item.event.event_id, error=str(e), now=current_time)
            stats["failed"] += 1

    stats["pending_remaining"] = active_queue.pending_count
    return stats


def drain_delivery_queue_async(
    queue: Optional[PersistentAlertQueue] = None,
    timeout: Optional[float] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """
    Executes process_delivery_queue asynchronously in a background thread
    so the detection loop is never blocked by retry attempts.
    """
    executor = _get_dispatch_executor()

    def _worker():
        res = process_delivery_queue(queue=queue, timeout=timeout)
        if callback:
            try:
                callback(res)
            except Exception:
                pass

    executor.submit(_worker)
