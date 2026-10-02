"""
=============================================================================
🐘 PROJECT ZOGAN — RISK ENGINE SIMULATION SCRIPT (PHASE 5)
=============================================================================

Demonstrates and verifies the rule-based risk engine behavior across a
simulated elephant approach trajectory towards a protected human settlement zone.

Usage:
    python ai/simulate_risk.py                    # Default approach sequence
    python ai/simulate_risk.py --scenario recede  # Receding trajectory
    python ai/simulate_risk.py --group-size 3     # Herd simulation
=============================================================================
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from ai.geofence import (
    ZONE_TYPE_PROTECTED,
    ZONE_TYPE_WARNING,
    ZONE_TYPE_MONITORING,
    ZONE_TYPE_OUTSIDE,
)
from ai.risk_engine import (
    RiskEngine,
    MovementTrendTracker,
    RISK_CRITICAL,
)
from alerts import create_alert_event, dispatch_alert


def run_simulation(
    scenario: str = "approach",
    group_size: int = 1,
    confidence: float = 0.95,
    log_events: bool = False,
    log_file: str = None,
):
    print("=" * 75)
    print("🐘 PROJECT ZOGAN — GEOFENCING & RISK SIMULATION (PHASE 5)")
    print(f"Scenario:    {scenario.upper()}")
    print(f"Group Size:  {group_size} elephant(s)")
    print(f"Confidence:  {int(confidence * 100)}%")
    print("Mode:        SOFTWARE SIMULATION (Demo)")
    print("=" * 75)
    print("⚠️  Notice: Decision-support prototype rules; not a validated biological model.\n")

    # Define trajectory sequences (distances to protected zone in meters)
    if scenario == "recede":
        distances = [150, 250, 400, 550, 700, 850, 1000, 1200]
    elif scenario == "stable":
        distances = [500, 505, 498, 502, 495, 501, 499]
    else:  # Default "approach"
        distances = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0]

    engine = RiskEngine()
    trend_tracker = MovementTrendTracker(
        window_size=config.TREND_WINDOW_SIZE,
        stability_threshold_meters=config.TREND_STABILITY_THRESHOLD_METERS,
    )

    print(
        f"{'STEP':<6} | {'DIST (m)':<9} | {'ZONE':<10} | {'TREND':<12} | {'SCORE':<7} | {'RISK LEVEL':<10} | {'ACTION'}"
    )
    print("-" * 75)

    history_records = []

    warning_boundary = config.BUFFER_RADIUS_METERS - config.VILLAGE_RADIUS_METERS
    forest_boundary = config.FOREST_RADIUS_METERS - config.VILLAGE_RADIUS_METERS

    for step, dist in enumerate(distances, start=1):
        # Determine zone by distance to protected zone (village boundary is at 0m)
        if dist <= 0:
            zone = ZONE_TYPE_PROTECTED
        elif dist <= warning_boundary:
            zone = ZONE_TYPE_WARNING
        elif dist <= forest_boundary:
            zone = ZONE_TYPE_MONITORING
        else:
            zone = ZONE_TYPE_OUTSIDE

        # Update approach trend
        trend = trend_tracker.add_observation(dist)

        # Evaluate risk score
        assessment = engine.evaluate(
            zone=zone,
            distance_to_protected=dist,
            trend=trend,
            group_size=group_size,
            confidence=confidence,
        )

        action = "🚨 TRIGGER ALERT" if assessment.alert_recommended else "🟢 MONITOR"
        if assessment.level == RISK_CRITICAL:
            action = "🚨 EMERGENCY ALARM"

        event_info = None
        if log_events and assessment.alert_recommended:
            ev = create_alert_event(
                confidence=confidence,
                risk_info=assessment.to_dict(),
                simulation_mode=True,
            )
            disp = dispatch_alert(ev, log_file=log_file, send_telegram=True, verbose=False)
            action += f" -> [LOGGED {ev.event_id}]"
            event_info = disp

        print(
            f"{step:<6} | {dist:<9.0f} | {zone:<10} | {trend:<12} | "
            f"{assessment.score:<7} | {assessment.level:<10} | {action}"
        )

        history_records.append(
            {
                "step": step,
                "dist": dist,
                "zone": zone,
                "trend": trend,
                "score": assessment.score,
                "level": assessment.level,
                "alert": assessment.alert_recommended,
                "dispatch": event_info,
            }
        )

    print("-" * 75)
    print("\n[OK] Simulation completed successfully.")
    return history_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Elephant Trajectory and Risk Assessment")
    parser.add_argument(
        "--scenario",
        choices=["approach", "recede", "stable"],
        default="approach",
        help="Trajectory scenario",
    )
    parser.add_argument("--group-size", type=int, default=1, help="Number of elephants in group")
    parser.add_argument("--conf", type=float, default=0.95, help="Detection confidence (0.0 - 1.0)")
    parser.add_argument("--log-events", action="store_true", help="Log alert events to JSONL during simulation")
    parser.add_argument("--log-file", type=str, default=None, help="Custom JSONL log file destination")
    args = parser.parse_args()

    run_simulation(
        scenario=args.scenario,
        group_size=args.group_size,
        confidence=args.conf,
        log_events=args.log_events,
        log_file=args.log_file,
    )
