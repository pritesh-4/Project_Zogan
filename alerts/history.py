"""
=============================================================================
🐘 PROJECT ZOGAN — ALERT EVENT HISTORY READER (PHASE 6)
=============================================================================

This module provides simple, robust utilities to inspect and query the
local alert event log (logs/alerts.jsonl):
  - get_recent_alerts() / load_alerts()
  - get_alert_count() / count_alerts()
  - get_latest_alert()
  - filter_by_risk()
  - Graceful handling when the log file does not exist (never crashes)
=============================================================================
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_LOG_FILE: str = "logs/alerts.jsonl"


def load_alerts(
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
    limit: Optional[int] = None,
    risk_level: Optional[str] = None,
    alert_level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Loads alert records from the JSONL log file.

    Parameters:
      log_file: Path to the JSONL log file.
      limit: If specified, returns the most recent 'limit' matching records.
      risk_level / alert_level: If specified, filters records by risk level (case-insensitive).

    Returns:
      List of alert record dictionaries, from oldest to newest.
      Returns [] if the file does not exist or has no valid entries.
    """
    path = Path(log_file)
    if not path.is_file():
        return []

    alerts: List[Dict[str, Any]] = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    if isinstance(record, dict):
                        alerts.append(record)
                except json.JSONDecodeError:
                    # Ignore corrupted / partial lines gracefully
                    continue
    except OSError:
        return []

    # Filter by risk/alert level if requested
    filter_lvl = alert_level if alert_level is not None else risk_level
    if filter_lvl is not None:
        target_lvl = str(filter_lvl).strip().upper()
        alerts = [a for a in alerts if str(a.get("alert_level") or a.get("risk_level", "")).upper() == target_lvl]

    # Limit to most recent records
    if limit is not None and limit > 0:
        return alerts[-limit:]

    return alerts


def get_recent_alerts(
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
    limit: Optional[int] = 10,
    risk_level: Optional[str] = None,
    alert_level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve recent alert events.
    """
    return load_alerts(log_file=log_file, limit=limit, risk_level=risk_level, alert_level=alert_level)


def count_alerts(
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
    risk_level: Optional[str] = None,
    alert_level: Optional[str] = None,
) -> int:
    """
    Returns the total count of alert events recorded, optionally filtered by risk level.
    """
    return len(load_alerts(log_file=log_file, risk_level=risk_level, alert_level=alert_level))


def get_alert_count(
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
    risk_level: Optional[str] = None,
    alert_level: Optional[str] = None,
) -> int:
    """
    Returns total count of stored alert events (alias for count_alerts).
    """
    return count_alerts(log_file=log_file, risk_level=risk_level, alert_level=alert_level)


def get_latest_alert(
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
    risk_level: Optional[str] = None,
    alert_level: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves the most recent recorded alert event, or None if no matching alerts exist.
    """
    matching = load_alerts(log_file=log_file, limit=1, risk_level=risk_level, alert_level=alert_level)
    return matching[-1] if matching else None


def filter_by_risk(
    alerts: List[Dict[str, Any]],
    risk_level: str,
) -> List[Dict[str, Any]]:
    """
    Filters a provided list of alert event dictionaries by their risk level.
    """
    target_lvl = str(risk_level).strip().upper()
    return [a for a in alerts if str(a.get("alert_level") or a.get("risk_level", "")).upper() == target_lvl]
