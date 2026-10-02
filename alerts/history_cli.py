"""
=============================================================================
🐘 PROJECT ZOGAN — ALERT HISTORY CLI (PHASE 6)
=============================================================================

Simple command-line utility to inspect recorded alert events from logs/alerts.jsonl.

Usage:
    python alerts/history_cli.py
    python alerts/history_cli.py --limit 5
    python alerts/history_cli.py --level HIGH
    python alerts/history_cli.py --log-file custom_alerts.jsonl
=============================================================================
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add parent directory to sys.path if executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alerts.history import DEFAULT_LOG_FILE, count_alerts, load_alerts


def display_history(log_file: str = DEFAULT_LOG_FILE, limit: int = 10, risk_level: str = None) -> None:
    """
    Displays stored alert incidents in a clean, human-readable terminal format.
    """
    print("=" * 60)
    print("🐘 PROJECT ZOGAN — ALERT HISTORY")
    print("=" * 60)

    total_count = count_alerts(log_file=log_file)
    matching_alerts = load_alerts(log_file=log_file, limit=limit, risk_level=risk_level)

    filter_info = f" [Filter: Level={risk_level.upper()}]" if risk_level else ""
    print(f"Log File:       {log_file}")
    print(f"Total Logged:   {total_count} event(s){filter_info}")
    print(f"Displaying:     {len(matching_alerts)} recent event(s)")
    print("-" * 60)

    if not matching_alerts:
        print(f"No alert records found in '{log_file}'.")
        print("=" * 60)
        return

    for idx, ev in enumerate(matching_alerts, start=1):
        ts = ev.get("timestamp", "N/A")
        eid = ev.get("event_id", "N/A")
        lvl = ev.get("alert_level") or ev.get("risk_level", "N/A")
        score = ev.get("risk_score")
        conf = ev.get("confidence")
        tid = ev.get("track_id")
        grp = ev.get("group_size")
        zone = ev.get("zone")
        dist = ev.get("distance_m")
        mov = ev.get("movement")
        sim = ev.get("simulation_mode") if "simulation_mode" in ev else ev.get("simulation")

        print(f"[{idx}] {ts} | ID: {eid}")
        score_str = f" (Score: {score}/100)" if score is not None else ""
        print(f"    Level:      {lvl}{score_str}")

        if conf is not None:
            conf_pct = int(round(conf * 100)) if conf <= 1.0 else int(round(conf))
            print(f"    Confidence: {conf_pct}%")
        if tid is not None:
            print(f"    Track:      #{tid}")
        if grp is not None:
            print(f"    Group:      {grp}")
        if zone is not None:
            print(f"    Zone:       {zone}")
        if dist is not None:
            print(f"    Distance:   {dist}m")
        if mov is not None:
            print(f"    Movement:   {mov}")
        if sim is not None:
            print(f"    Mode:       {'SIMULATION' if sim else 'LIVE'}")

        print("-" * 60)

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View Project Zogan Alert Incident History")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of recent alert records to display (default: 10)",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        help="Filter by alert level (LOW, MEDIUM, HIGH, CRITICAL)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help=f"Path to JSONL log file (default: {DEFAULT_LOG_FILE})",
    )
    args = parser.parse_args()

    display_history(log_file=args.log_file, limit=args.limit, risk_level=args.level)
