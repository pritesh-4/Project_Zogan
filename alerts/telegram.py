"""
=============================================================================
🐘 PROJECT ZOGAN — TELEGRAM REMOTE ALERT SERVICE (PHASE 6)
=============================================================================

This module provides remote incident alert delivery via the Telegram Bot API.

Security & Architecture Rules:
  - Credentials read strictly from environment variables:
      TELEGRAM_BOT_TOKEN
      TELEGRAM_CHAT_ID
  - Automatically loads .env file if available (never committed to git).
  - NEVER hardcodes tokens or chat IDs.
  - Formats clear, readable alert messages with ONLY existing values.
  - Fail-safe exception handling: Telegram downtime, network drops, or missing
    credentials will NEVER crash the video detection pipeline.
=============================================================================
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from alerts.models import AlertEvent

# Environment variable keys
ENV_BOT_TOKEN: str = "TELEGRAM_BOT_TOKEN"
ENV_CHAT_ID: str = "TELEGRAM_CHAT_ID"

# Telegram Bot API Endpoint base URL
TELEGRAM_API_URL_TEMPLATE: str = "https://api.telegram.org/bot{token}/sendMessage"


def load_env(env_file: Union[str, Path] = ".env") -> None:
    """
    Safely loads environment variables from a .env file without overwriting
    pre-existing environment variables.
    """
    path = Path(env_file)
    if not path.is_file():
        return

    # Try python-dotenv if available
    try:
        import importlib

        dotenv = importlib.import_module("dotenv")
        dotenv.load_dotenv(dotenv_path=path, override=False)
        return
    except Exception:
        pass

    # Built-in fallback .env reader
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                if "=" in line_str:
                    key, val = line_str.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass


# Attempt automatic .env loading upon module import
load_env()


def get_telegram_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieves TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment.

    Returns:
      (bot_token, chat_id) or (None, None) if not configured.
    """
    token = os.getenv(ENV_BOT_TOKEN)
    chat_id = os.getenv(ENV_CHAT_ID)

    clean_token = token.strip() if token else None
    clean_chat_id = chat_id.strip() if chat_id else None

    # Filter out empty strings or template placeholders
    if clean_token in ("", "YOUR_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE"):
        clean_token = None
    if clean_chat_id in ("", "YOUR_CHAT_ID", "YOUR_CHAT_ID_HERE"):
        clean_chat_id = None

    return clean_token, clean_chat_id


def is_telegram_configured() -> bool:
    """
    Returns True if valid non-empty Telegram credentials are configured in the environment.
    """
    token, chat_id = get_telegram_credentials()
    return bool(token and chat_id)


def format_telegram_message(event: Union[AlertEvent, Dict[str, Any]]) -> str:
    """
    Formats a clean, human-readable Telegram alert message.
    Only includes values that actually exist (do not fabricate values).

    Target Format:
      🚨 PROJECT ZOGAN ALERT

      Risk: HIGH
      Score: 72
      Confidence: 94%

      Track: #2
      Group Size: 4

      Zone: WARNING
      Distance: 380 m
      Movement: APPROACHING

      Time: 2026-10-03 ...

      Mode: SIMULATION
    """
    if isinstance(event, AlertEvent):
        data = event.to_dict(exclude_none=True)
    elif isinstance(event, dict):
        data = {k: v for k, v in event.items() if v is not None}
    else:
        raise TypeError(f"Expected AlertEvent or dict, received: {type(event)}")

    blocks = []

    # Title Banner
    blocks.append("🚨 PROJECT ZOGAN ALERT")

    # Risk Metrics Section
    risk_lines = []
    if "risk_level" in data:
        risk_lines.append(f"Risk: {data['risk_level']}")
    if "risk_score" in data:
        risk_lines.append(f"Score: {data['risk_score']}")
    if "confidence" in data:
        c = data["confidence"]
        if isinstance(c, (int, float)):
            pct = int(round(c * 100)) if c <= 1.0 else int(round(c))
            risk_lines.append(f"Confidence: {pct}%")
    if risk_lines:
        blocks.append("\n".join(risk_lines))

    # Tracking Section
    track_lines = []
    if "track_id" in data:
        track_lines.append(f"Track: #{data['track_id']}")
    if "group_size" in data:
        track_lines.append(f"Group Size: {data['group_size']}")
    if track_lines:
        blocks.append("\n".join(track_lines))

    # Geofence & Movement Section
    geo_lines = []
    if "zone" in data:
        geo_lines.append(f"Zone: {data['zone']}")
    if "distance_m" in data:
        d = data["distance_m"]
        if isinstance(d, (int, float)):
            geo_lines.append(f"Distance: {int(round(d))} m")
        else:
            geo_lines.append(f"Distance: {d}")
    if "movement" in data:
        geo_lines.append(f"Movement: {data['movement']}")
    if geo_lines:
        blocks.append("\n".join(geo_lines))

    # Reasons / Explainability Section (if available)
    if "reasons" in data and data["reasons"]:
        reasons_list = data["reasons"]
        if isinstance(reasons_list, list):
            reason_lines = [f"• {r}" for r in reasons_list[:3]]
            blocks.append("Assessment Reasons:\n" + "\n".join(reason_lines))

    # Timestamp Section
    if "timestamp" in data:
        blocks.append(f"Time: {data['timestamp']}")

    # System Mode Section
    if "simulation" in data:
        sim = data["simulation"]
        mode_str = "SIMULATION" if sim else "LIVE"
        blocks.append(f"Mode: {mode_str}")

    return "\n\n".join(blocks)


def send_telegram_message(
    message: str,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Sends a text message via the Telegram Bot API (sendMessage).
    Never raises an unhandled exception so the detection pipeline remains unblocked.

    Returns:
      Result dictionary containing:
        - success: bool
        - status: 'sent', 'not_configured', 'http_error', 'network_error', 'api_error', 'timeout', etc.
        - error: Optional error description string
        - response: Optional decoded Telegram API response
    """
    if not token or not chat_id:
        env_token, env_chat_id = get_telegram_credentials()
        token = token or env_token
        chat_id = chat_id or env_chat_id

    if not token or not chat_id:
        return {
            "success": False,
            "status": "not_configured",
            "error": "Telegram credentials missing: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.",
        }

    url = TELEGRAM_API_URL_TEMPLATE.format(token=token)
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            body_bytes = response.read()
            body_data = json.loads(body_bytes.decode("utf-8"))

            if status_code == 200 and body_data.get("ok"):
                return {
                    "success": True,
                    "status": "sent",
                    "response": body_data,
                }
            else:
                desc = body_data.get("description", "Unknown Telegram API response")
                return {
                    "success": False,
                    "status": "api_error",
                    "error": desc,
                    "response": body_data,
                }

    except urllib.error.HTTPError as e:
        error_msg = f"HTTP Error {e.code}: {e.reason}"
        try:
            err_json = json.loads(e.read().decode("utf-8"))
            if "description" in err_json:
                error_msg += f" - {err_json['description']}"
        except Exception:
            pass
        return {"success": False, "status": "http_error", "error": error_msg}

    except urllib.error.URLError as e:
        return {"success": False, "status": "network_error", "error": f"Network Error: {e.reason}"}

    except TimeoutError:
        return {"success": False, "status": "timeout", "error": f"Request timed out after {timeout} seconds"}

    except Exception as e:
        return {"success": False, "status": "unexpected_error", "error": str(e)}


def send_telegram_alert(
    event: Union[AlertEvent, Dict[str, Any]],
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Formats the given alert event and delivers it via Telegram.
    Guaranteed fail-safe; never crashes the caller.
    """
    try:
        msg = format_telegram_message(event)
        return send_telegram_message(msg, token=token, chat_id=chat_id, timeout=timeout)
    except Exception as e:
        return {
            "success": False,
            "status": "format_error",
            "error": f"Failed to format or send Telegram alert: {e}",
        }
