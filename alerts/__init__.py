"""
=============================================================================
🐘 PROJECT ZOGAN — ALERTS PACKAGE (PHASE 6)
=============================================================================

Centralized event logging, alert history querying, and remote notification
services for Project Zogan.
=============================================================================
"""

from alerts.dispatcher import dispatch_alert
from alerts.event_logger import AlertEventLogger, log_alert
from alerts.history import (
    count_alerts,
    filter_by_risk,
    get_alert_count,
    get_latest_alert,
    get_recent_alerts,
    load_alerts,
)
from alerts.models import AlertEvent, create_alert_event
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
]
