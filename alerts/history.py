"""
=============================================================================
🐘 PROJECT ZOGAN — ALERT EVENT HISTORY READER (PHASE 6)
=============================================================================

This module provides a simple, robust utility to inspect and query the
local alert event log (logs/alerts.jsonl):
  - Load recent alert records
  - Count stored alerts (optionally filtered by risk level)
  - Retrieve the latest alert record
  - Filter alerts by risk level (e.g. HIGH, CRITICAL, MEDIUM, LOW)
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
) -> List[Dict[str, Any]]:
    """
    Loads alert records from the JSONL log file.

    Parameters:
      log_file: Path to the JSONL log file.
      limit: If specified, returns the most recent 'limit' matching records.
      risk_level: If specified, filters records by risk level (case-insensitive).

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

    # Filter by risk level if requested
    if risk_level is not None:
        target_lvl = str(risk_level).strip().upper()
        alerts = [a for a in alerts if str(a.get("risk_level", "")).upper() == target_lvl]

    # Limit to most recent records
    if limit is not None and limit > 0:
        return alerts[-limit:]

    return alerts


def count_alerts(
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
    risk_level: Optional[str] = None,
) -> int:
    """
    Returns the total count of alert events recorded, optionally filtered by risk level.
    """
    return len(load_alerts(log_file=log_file, risk_level=risk_level))


def get_latest_alert(
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
    risk_level: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves the most recent recorded alert event, or None if no matching alerts exist.
    """
    matching = load_alerts(log_file=log_file, limit=1, risk_level=risk_level)
    return matching[-1] if matching else None


def filter_by_risk(
    alerts: List[Dict[str, Any]],
    risk_level: str,
) -> List[Dict[str, Any]]:
    """
    Filters a provided list of alert event dictionaries by their risk level.
    """
    target_lvl = str(risk_level).strip().upper()
    return [a for a in alerts if str(a.get("risk_level", "")).upper() == target_lvl]
