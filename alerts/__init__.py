"""
=============================================================================
🐘 PROJECT ZOGAN — ALERTS PACKAGE (PHASE 6 & PHASE 9)
=============================================================================

Centralized event logging, alert history querying, persistent offline queue,
and remote notification services for Project Zogan.
=============================================================================
"""

from alerts.dispatcher import (
    dispatch_alert,
    drain_delivery_queue_async,
    get_alert_queue,
    process_delivery_queue,
    set_alert_queue,
)
from alerts.event_logger import AlertEventLogger, log_alert
from alerts.history import (
    count_alerts,
    filter_by_risk,
    get_alert_count,
    get_latest_alert,
    get_recent_alerts,
    load_alerts,
)
from alerts.models import (
    DELIVERY_FAILED,
    DELIVERY_NOT_CONFIGURED,
    DELIVERY_PENDING,
    DELIVERY_RETRYING,
    DELIVERY_SENT,
    DELIVERY_SKIPPED,
    AlertEvent,
    create_alert_event,
)
from alerts.queue import PersistentAlertQueue, QueueItem
from alerts.telegram import (
    format_telegram_message,
    get_telegram_credentials,
    is_telegram_configured,
    send_telegram_alert,
    send_telegram_message,
)

__all__ = [
    "AlertEvent",
    "create_alert_event",
    "AlertEventLogger",
    "log_alert",
    "load_alerts",
    "get_recent_alerts",
    "count_alerts",
    "get_alert_count",
    "get_latest_alert",
    "filter_by_risk",
    "format_telegram_message",
    "get_telegram_credentials",
    "is_telegram_configured",
    "send_telegram_message",
    "send_telegram_alert",
    "dispatch_alert",
    "drain_delivery_queue_async",
    "get_alert_queue",
    "set_alert_queue",
    "process_delivery_queue",
    "PersistentAlertQueue",
    "QueueItem",
    "DELIVERY_PENDING",
    "DELIVERY_SENT",
    "DELIVERY_FAILED",
    "DELIVERY_RETRYING",
    "DELIVERY_NOT_CONFIGURED",
    "DELIVERY_SKIPPED",
]
