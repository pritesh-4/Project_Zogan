"""
=============================================================================
🐘 PROJECT ZOGAN — ALERT EVENT LOGGER (PHASE 6)
=============================================================================

This module provides persistent local logging of confirmed alert events
in JSON Lines (JSONL) format to:
  logs/alerts.jsonl

Safety & Reliability Requirements:
  - Automatically creates the logs/ directory if missing (safe against missing dirs).
  - Encodes each event as a single-line JSON string appended atomically.
  - Excludes non-existent/None attributes to prevent fabricated data.
  - Fail-safe exception handling so log failures never crash detection.
=============================================================================
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from alerts.models import AlertEvent

DEFAULT_LOG_FILE: str = "logs/alerts.jsonl"


class AlertEventLogger:
    """
    Manages local persistent recording of alert events in JSON Lines format.
    """

    def __init__(self, log_file: Union[str, Path] = DEFAULT_LOG_FILE):
        self.log_file = Path(log_file)

    def _ensure_directory(self) -> None:
        """
        Safely creates the destination log directory if it does not exist.
        """
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[EVENT LOGGER WARNING] Could not create directory '{self.log_file.parent}': {e}")

    def log(self, event: Union[AlertEvent, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Appends a confirmed alert event to the JSONL log file.

        Inputs:
          event: AlertEvent object or equivalent dictionary

        Returns:
          Dictionary representing the serialized event data that was logged.
        """
        self._ensure_directory()

        if isinstance(event, AlertEvent):
            event_dict = event.to_dict(exclude_none=True)
        elif isinstance(event, dict):
            event_dict = {k: v for k, v in event.items() if v is not None}
        else:
            raise TypeError(f"Expected AlertEvent or dict, received: {type(event)}")

        line = json.dumps(event_dict, ensure_ascii=False)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

        return event_dict


def log_alert(
    event: Union[AlertEvent, Dict[str, Any]],
    log_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to log an alert event using AlertEventLogger.
    """
    logger = AlertEventLogger(log_file=log_file or DEFAULT_LOG_FILE)
    return logger.log(event)
