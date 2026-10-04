"""
=============================================================================
🐘 PROJECT ZOGAN — SYSTEM HEALTH CHECK
=============================================================================

Lightweight startup diagnostic that verifies all system components are
available and correctly configured. Reports REAL states — no fake "OK".

Usage:
    python scripts/health_check.py
    python -m scripts.health_check
=============================================================================
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_python() -> str:
    v = sys.version_info
    return f"OK ({v.major}.{v.minor}.{v.micro})"


def check_opencv() -> str:
    try:
        import cv2

        return f"OK ({cv2.__version__})"
    except ImportError:
        return "MISSING (pip install opencv-python)"


def check_ultralytics() -> str:
    try:
        import ultralytics

        return f"OK ({ultralytics.__version__})"
    except ImportError:
        return "MISSING (pip install ultralytics)"


def check_torch() -> str:
    try:
        import torch

        gpu = "GPU" if torch.cuda.is_available() else "CPU"
        return f"OK ({torch.__version__}, {gpu})"
    except ImportError:
        return "MISSING (pip install torch)"


def check_model() -> str:
    try:
        import config
        from ai.model_manager import resolve_model_path

        path, label = resolve_model_path(use_custom=config.USE_CUSTOM_MODEL)
        if path and Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            return f"OK ({label}, {size_mb:.1f} MB)"
        return f"NOT FOUND ({label})"
    except Exception as e:
        return f"ERROR ({e})"


def check_model_class() -> str:
    try:
        import config
        from ai.model_manager import resolve_model_path

        path, _ = resolve_model_path(use_custom=config.USE_CUSTOM_MODEL)
        if not path or not Path(path).exists():
            return "SKIPPED (no model)"

        from ultralytics import YOLO

        model = YOLO(path)
        names = model.names
        for cid, cname in names.items():
            if cname.lower() == config.TARGET_CLASS.lower():
                return f"OK ({cname}, class_id={cid})"
        return f"NOT FOUND (available: {list(names.values())})"
    except Exception as e:
        return f"ERROR ({e})"


def check_config() -> str:
    try:
        import config

        # Validate critical thresholds
        errors = []
        if not (0.0 < config.CONFIDENCE_THRESHOLD <= 1.0):
            errors.append(f"CONFIDENCE_THRESHOLD={config.CONFIDENCE_THRESHOLD}")
        if config.REQUIRED_DETECTIONS < 1:
            errors.append(f"REQUIRED_DETECTIONS={config.REQUIRED_DETECTIONS}")
        if config.ALERT_COOLDOWN_SECONDS < 0:
            errors.append(f"ALERT_COOLDOWN_SECONDS={config.ALERT_COOLDOWN_SECONDS}")
        if config.VILLAGE_RADIUS_METERS <= 0:
            errors.append(f"VILLAGE_RADIUS_METERS={config.VILLAGE_RADIUS_METERS}")

        if errors:
            return f"INVALID ({', '.join(errors)})"
        return "OK"
    except Exception as e:
        return f"ERROR ({e})"


def check_dataset() -> str:
    try:
        data_yaml = Path("datasets/elephant/data.yaml")
        if not data_yaml.exists():
            return "NOT FOUND"

        train_dir = Path("datasets/elephant/images/train")
        val_dir = Path("datasets/elephant/images/val")
        test_dir = Path("datasets/elephant/images/test")

        counts = []
        for name, d in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
            if d.exists():
                n = len(list(d.glob("*.jpg")))
                counts.append(f"{name}={n}")
            else:
                counts.append(f"{name}=MISSING")

        return f"AVAILABLE ({', '.join(counts)})"
    except Exception as e:
        return f"ERROR ({e})"


def check_telegram() -> str:
    try:
        from alerts.telegram import is_telegram_configured

        if is_telegram_configured():
            return "ENABLED (credentials configured)"
        return "DISABLED (no credentials in environment)"
    except Exception as e:
        return f"ERROR ({e})"


def check_gps() -> str:
    import config

    if config.DEFAULT_SIMULATION_MODE:
        return f"SIMULATION ({config.CAMERA_LATITUDE:.6f}, {config.CAMERA_LONGITUDE:.6f})"
    return "UNAVAILABLE (no hardware GPS)"


def check_risk_engine() -> str:
    try:
        from ai.geofence import ZONE_TYPE_WARNING
        from ai.risk_engine import RiskEngine, TREND_APPROACHING

        engine = RiskEngine()
        assessment = engine.evaluate(ZONE_TYPE_WARNING, 350.0, TREND_APPROACHING, 1, 0.90, persistence_frames=5)
        return f"OK (Score={assessment.score}, Level={assessment.level})"
    except Exception as e:
        return f"ERROR ({e})"


def check_health_monitor() -> str:
    try:
        from monitoring import SystemHealthMonitor

        monitor = SystemHealthMonitor()
        monitor.record_camera_open(True)
        monitor.record_detector_status("READY")
        monitor.record_tracker_status("READY")
        monitor.record_risk_engine_status("READY")
        monitor.record_frame_read(True)
        snapshot = monitor.evaluate()
        return f"OK (Status={snapshot.overall_status})"
    except Exception as e:
        return f"ERROR ({e})"


def check_offline_queue() -> str:
    try:
        import config
        from alerts.queue import PersistentAlertQueue

        if not getattr(config, "OFFLINE_QUEUE_ENABLED", True):
            return "DISABLED (disabled in config)"
        queue = PersistentAlertQueue()
        pending = queue.pending_count
        return f"OK ({pending} pending, max={getattr(config, 'ALERT_QUEUE_MAX_SIZE', 100)})"
    except Exception as e:
        return f"ERROR ({e})"


def run_health_check() -> bool:
    """Runs all health checks and prints a formatted report."""
    print()
    print("PROJECT ZOGAN SYSTEM CHECK")
    print("=" * 50)

    checks = [
        ("Python", check_python),
        ("OpenCV", check_opencv),
        ("Ultralytics", check_ultralytics),
        ("PyTorch", check_torch),
        ("Model", check_model),
        ("Model Class", check_model_class),
        ("Config", check_config),
        ("Dataset", check_dataset),
        ("Risk Engine", check_risk_engine),
        ("Health Monitor", check_health_monitor),
        ("Offline Queue", check_offline_queue),
        ("Telegram", check_telegram),
        ("GPS", check_gps),
    ]

    all_ok = True
    for name, check_fn in checks:
        try:
            result = check_fn()
        except Exception as e:
            result = f"ERROR ({e})"

        status_ok = (
            result.startswith("OK")
            or result.startswith("AVAILABLE")
            or result.startswith("SIMULATION")
            or result.startswith("DISABLED")
            or result.startswith("ENABLED")
        )

        marker = "✓" if status_ok else "✗"
        if not status_ok and not result.startswith("DISABLED") and not result.startswith("SIMULATION"):
            all_ok = False

        print(f"  {marker} {name:15s} {result}")

    print("=" * 50)
    if all_ok:
        print("STATUS: ALL CHECKS PASSED")
    else:
        print("STATUS: ISSUES DETECTED (see above)")
    print()

    return all_ok


if __name__ == "__main__":
    success = run_health_check()
    sys.exit(0 if success else 1)
