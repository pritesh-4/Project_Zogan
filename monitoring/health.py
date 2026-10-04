"""
=============================================================================
🐘 PROJECT ZOGAN — SYSTEM HEALTH MONITORING & OBSERVABILITY (PHASE 8)
=============================================================================

This module provides an explainable, deterministic health monitoring layer
for the Project Zogan wildlife monitoring pipeline:
  - Input stream status & frame freshness tracking
  - Rolling FPS throughput monitoring
  - Component lifecycle status (Detector, Tracker, Risk Engine)
  - Bounded memory error tracking & sanitization
  - Centralized multi-signal health evaluation & state transitions
  - Clear separation between System Health (is Zogan running?) and
    Threat Risk (how dangerous is the elephant situation?)
=============================================================================
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import config
from monitoring.metrics import RollingFPS, RuntimeMetrics

# =============================================================================
# 🏷️ STATUS CONSTANTS
# =============================================================================

# Overall System Status
HEALTH_HEALTHY: str = "HEALTHY"
HEALTH_DEGRADED: str = "DEGRADED"
HEALTH_OFFLINE: str = "OFFLINE"
HEALTH_UNKNOWN: str = "UNKNOWN"

# Input / Camera Stream Status
INPUT_ONLINE: str = "ONLINE"
INPUT_DEGRADED: str = "DEGRADED"
INPUT_LOST: str = "LOST"
INPUT_OFFLINE: str = "OFFLINE"
INPUT_UNKNOWN: str = "UNKNOWN"

# Component / Subsystem Status
SUBSYSTEM_READY: str = "READY"
SUBSYSTEM_DEGRADED: str = "DEGRADED"
SUBSYSTEM_FAILED: str = "FAILED"
SUBSYSTEM_UNKNOWN: str = "UNKNOWN"

# Network & Remote Delivery Status (Phase 9)
NETWORK_AVAILABLE: str = "AVAILABLE"
NETWORK_UNAVAILABLE: str = "UNAVAILABLE"
NETWORK_UNKNOWN: str = "UNKNOWN"

ALERT_DELIVERY_HEALTHY: str = "HEALTHY"
ALERT_DELIVERY_DEGRADED: str = "DEGRADED"
ALERT_DELIVERY_FAILED: str = "FAILED"


# =============================================================================
# 📦 HEALTH SNAPSHOT DATA MODEL
# =============================================================================


@dataclass
class HealthSnapshot:
    """
    Structured, serializable representation of complete system health.
    """

    overall_status: str
    input_status: str
    detector_status: str
    tracker_status: str
    risk_engine_status: str
    current_fps: float
    average_fps: float
    last_frame_age_seconds: float
    uptime_seconds: float
    total_frames: int
    processed_frames: int
    dropped_frames: int
    error_count: int
    consecutive_errors: int
    network_status: str = NETWORK_UNKNOWN
    alert_delivery_status: str = ALERT_DELIVERY_HEALTHY
    pending_alert_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the health snapshot to a JSON-compatible dictionary."""
        age_val = round(self.last_frame_age_seconds, 3) if self.last_frame_age_seconds != float("inf") else None
        return {
            "overall_status": self.overall_status,
            "input_status": self.input_status,
            "detector_status": self.detector_status,
            "tracker_status": self.tracker_status,
            "risk_engine_status": self.risk_engine_status,
            "network_status": self.network_status,
            "alert_delivery_status": self.alert_delivery_status,
            "pending_alert_count": self.pending_alert_count,
            "current_fps": round(self.current_fps, 2),
            "average_fps": round(self.average_fps, 2),
            "last_frame_age_seconds": age_val,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "dropped_frames": self.dropped_frames,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "reasons": list(self.reasons),
            "timestamp": self.timestamp,
        }


# =============================================================================
# 🩺 SYSTEM HEALTH MONITOR
# =============================================================================


class SystemHealthMonitor:
    """
    Central observability and health evaluation monitor for Project Zogan.

    Aggregates camera/input signals, FPS throughput, component health,
    and runtime errors into a deterministic overall status.
    """

    def __init__(
        self,
        fps_minimum: Optional[float] = None,
        fps_window_size: Optional[int] = None,
        freshness_degraded_seconds: Optional[float] = None,
        freshness_offline_seconds: Optional[float] = None,
        max_consecutive_frame_failures: Optional[int] = None,
        max_consecutive_frame_failures_offline: Optional[int] = None,
        max_consecutive_errors_degraded: Optional[int] = None,
        max_consecutive_errors_offline: Optional[int] = None,
        max_error_history: Optional[int] = None,
        heartbeat_timeout_seconds: Optional[float] = None,
        start_time: Optional[float] = None,
    ):
        # Configuration with fallback to centralized settings
        self.fps_minimum: float = (
            fps_minimum if fps_minimum is not None else getattr(config, "HEALTH_FPS_MINIMUM_HEALTHY", 12.0)
        )
        self.fps_window_size: int = (
            fps_window_size if fps_window_size is not None else getattr(config, "HEALTH_FPS_WINDOW_SIZE", 30)
        )
        self.freshness_degraded_seconds: float = (
            freshness_degraded_seconds
            if freshness_degraded_seconds is not None
            else getattr(config, "HEALTH_FRESHNESS_DEGRADED_SECONDS", 1.5)
        )
        self.freshness_offline_seconds: float = (
            freshness_offline_seconds
            if freshness_offline_seconds is not None
            else getattr(config, "HEALTH_FRESHNESS_OFFLINE_SECONDS", 5.0)
        )
        self.max_consecutive_frame_failures: int = (
            max_consecutive_frame_failures
            if max_consecutive_frame_failures is not None
            else getattr(config, "HEALTH_MAX_CONSECUTIVE_FRAME_FAILURES", 5)
        )
        self.max_consecutive_frame_failures_offline: int = (
            max_consecutive_frame_failures_offline
            if max_consecutive_frame_failures_offline is not None
            else getattr(config, "HEALTH_MAX_CONSECUTIVE_FRAME_FAILURES_OFFLINE", 20)
        )
        self.max_consecutive_errors_degraded: int = (
            max_consecutive_errors_degraded
            if max_consecutive_errors_degraded is not None
            else getattr(config, "HEALTH_MAX_CONSECUTIVE_ERRORS_DEGRADED", 3)
        )
        self.max_consecutive_errors_offline: int = (
            max_consecutive_errors_offline
            if max_consecutive_errors_offline is not None
            else getattr(config, "HEALTH_MAX_CONSECUTIVE_ERRORS_OFFLINE", 10)
        )
        self.max_error_history: int = (
            max_error_history if max_error_history is not None else getattr(config, "HEALTH_MAX_ERROR_HISTORY", 20)
        )
        self.heartbeat_timeout_seconds: float = (
            heartbeat_timeout_seconds
            if heartbeat_timeout_seconds is not None
            else getattr(config, "HEALTH_HEARTBEAT_TIMEOUT_SECONDS", 5.0)
        )

        init_time = time.time() if start_time is None else start_time
        self._metrics = RuntimeMetrics(start_time=init_time)
        self._rolling_fps = RollingFPS(window_size=self.fps_window_size)

        # Subsystem states
        self._camera_opened: Optional[bool] = None
        self._detector_status: str = SUBSYSTEM_UNKNOWN
        self._tracker_status: str = SUBSYSTEM_UNKNOWN
        self._risk_engine_status: str = SUBSYSTEM_UNKNOWN
        self._network_status: str = NETWORK_UNKNOWN
        self._alert_delivery_status: str = ALERT_DELIVERY_HEALTHY
        self._pending_alert_count: int = 0

        # Error tracking
        self._last_error_message: Optional[str] = None
        self._last_error_time: Optional[str] = None
        self._error_history: deque = deque(maxlen=self.max_error_history)

        # Transition tracking
        self._last_overall_status: Optional[str] = None

    # =========================================================================
    # 📥 INPUT & LIFECYCLE RECORDING API
    # =========================================================================

    def record_camera_open(
        self,
        success: bool,
        source_info: str = "",
        now: Optional[float] = None,
    ) -> None:
        """Records initial camera/video stream open status."""
        current_time = time.time() if now is None else now
        self._camera_opened = success
        if success:
            self._metrics.consecutive_dropped_frames = 0
        else:
            self.record_error(
                component="camera",
                error=f"Failed to open video source: {source_info}",
                fatal=True,
                now=current_time,
            )

    def record_frame_read(
        self,
        success: bool,
        now: Optional[float] = None,
    ) -> None:
        """
        Records the outcome of an attempted frame read from the video stream.
        """
        current_time = time.time() if now is None else now
        self._metrics.total_frames += 1

        if success:
            self._metrics.last_frame_time = current_time
            self._metrics.consecutive_dropped_frames = 0
            self._rolling_fps.update(current_time)
        else:
            self._metrics.dropped_frames += 1
            self._metrics.consecutive_dropped_frames += 1

    def record_frame_processed(
        self,
        detection_count: int = 0,
        active_tracks: int = 0,
        now: Optional[float] = None,
    ) -> None:
        """
        Records successful completion of detection and risk processing for a frame.
        """
        current_time = time.time() if now is None else now
        self._metrics.processed_frames += 1
        self._metrics.last_processed_time = current_time
        self._metrics.detection_count += detection_count
        self._metrics.active_tracks = active_tracks

        # Successful processing decays consecutive errors
        if self._metrics.consecutive_errors > 0:
            self._metrics.consecutive_errors = max(0, self._metrics.consecutive_errors - 1)

    def record_detector_status(
        self,
        status: str,
        error_message: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        """Records detector component status."""
        self._detector_status = status
        if error_message:
            self.record_error(
                component="detector",
                error=error_message,
                fatal=(status == SUBSYSTEM_FAILED),
                now=now,
            )

    def record_tracker_status(
        self,
        status: str,
        error_message: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        """Records multi-object tracker component status."""
        self._tracker_status = status
        if error_message:
            self.record_error(
                component="tracker",
                error=error_message,
                fatal=(status == SUBSYSTEM_FAILED),
                now=now,
            )

    def record_risk_engine_status(
        self,
        status: str,
        error_message: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        """Records risk assessment engine component status."""
        self._risk_engine_status = status
        if error_message:
            self.record_error(
                component="risk_engine",
                error=error_message,
                fatal=(status == SUBSYSTEM_FAILED),
                now=now,
            )

    def record_error(
        self,
        component: str,
        error: Union[str, Exception],
        fatal: bool = False,
        now: Optional[float] = None,
    ) -> None:
        """
        Records a sanitized, bounded error occurrence.
        Never logs secrets, tokens, or credentials.
        """
        current_time = time.time() if now is None else now
        raw_msg = str(error)

        # Sanitize against accidental token/credential leaks
        sanitized = raw_msg
        for secret_cue in ("bot", "token", "password", "key", "secret"):
            if secret_cue in sanitized.lower() and "=" in sanitized:
                sanitized = "[REDACTED_CREDENTIAL]"
                break

        msg = f"[{component.upper()}] {sanitized}"
        time_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time))

        self._metrics.error_count += 1
        self._metrics.consecutive_errors += 1
        self._last_error_message = msg
        self._last_error_time = time_str
        self._error_history.append((current_time, component, sanitized))

    def record_delivery_result(
        self,
        success: bool,
        error: Optional[str] = None,
        pending_count: int = 0,
        now: Optional[float] = None,
    ) -> None:
        """
        Records the outcome of a remote alert delivery attempt.
        Infers network availability naturally from real delivery results.
        """
        current_time = time.time() if now is None else now
        self._pending_alert_count = pending_count
        if success:
            self._network_status = NETWORK_AVAILABLE
            self._alert_delivery_status = ALERT_DELIVERY_HEALTHY if pending_count == 0 else ALERT_DELIVERY_DEGRADED
        else:
            self._network_status = NETWORK_UNAVAILABLE
            self._alert_delivery_status = ALERT_DELIVERY_DEGRADED
            if error:
                self.record_error("network", error, fatal=False, now=current_time)

    def update_pending_alerts(self, count: int) -> None:
        """Updates active count of alerts awaiting retry in the offline queue."""
        self._pending_alert_count = count
        if count > 0 and self._alert_delivery_status == ALERT_DELIVERY_HEALTHY:
            self._alert_delivery_status = ALERT_DELIVERY_DEGRADED

    def record_heartbeat(self, now: Optional[float] = None) -> None:
        """Records a pulse proving the monitoring pipeline is alive."""
        current_time = time.time() if now is None else now
        self._metrics.last_heartbeat_time = current_time

    def is_heartbeat_alive(self, now: Optional[float] = None) -> bool:
        """Checks if the heartbeat is within the timeout window."""
        if self._metrics.last_heartbeat_time <= 0.0:
            return True
        current_time = time.time() if now is None else now
        return (current_time - self._metrics.last_heartbeat_time) <= self.heartbeat_timeout_seconds

    # =========================================================================
    # 🔍 CENTRAL HEALTH EVALUATION
    # =========================================================================

    def evaluate(self, now: Optional[float] = None) -> HealthSnapshot:
        """
        Evaluates all input signals and component statuses to compute
        a deterministic HealthSnapshot.

        Explainable reasons are generated for any degraded or offline state.
        """
        current_time = time.time() if now is None else now
        reasons: List[str] = []

        # 1. Evaluate Camera / Input Stream Status
        if self._camera_opened is False:
            input_status = INPUT_OFFLINE
            reasons.append("Camera / input stream failed to open")
        elif self._metrics.total_frames == 0:
            input_status = INPUT_UNKNOWN
        else:
            if self._metrics.last_frame_time is None:
                frame_age = self._metrics.uptime(current_time)
            else:
                frame_age = self._metrics.last_frame_age(current_time)
            consec_drops = self._metrics.consecutive_dropped_frames

            if (
                consec_drops >= self.max_consecutive_frame_failures_offline
                or frame_age >= self.freshness_offline_seconds
            ):
                input_status = INPUT_OFFLINE
                reasons.append(
                    f"Input stream lost / offline (frame age: {frame_age:.1f}s, consecutive drops: {consec_drops})"
                )
            elif consec_drops >= self.max_consecutive_frame_failures or frame_age >= self.freshness_degraded_seconds:
                input_status = INPUT_DEGRADED
                reasons.append(
                    f"Input stream degraded (frame age: {frame_age:.1f}s, consecutive drops: {consec_drops})"
                )
            else:
                input_status = INPUT_ONLINE

        # 2. Evaluate Component Subsystem Statuses
        if self._detector_status == SUBSYSTEM_FAILED:
            reasons.append("YOLO elephant detector failed or unavailable")
        elif self._detector_status == SUBSYSTEM_DEGRADED:
            reasons.append("YOLO elephant detector running in degraded state")

        if self._tracker_status == SUBSYSTEM_FAILED:
            reasons.append("Multi-object tracking engine failed")
        elif self._tracker_status == SUBSYSTEM_DEGRADED:
            reasons.append("Multi-object tracking engine degraded")

        if self._risk_engine_status == SUBSYSTEM_FAILED:
            reasons.append("Spatial risk assessment engine failed")
        elif self._risk_engine_status == SUBSYSTEM_DEGRADED:
            reasons.append("Spatial risk assessment engine degraded")

        # 3. Evaluate FPS Throughput
        current_fps = self._rolling_fps.current_fps
        # Only evaluate FPS once we have processed a sufficient sample window
        fps_evaluated = input_status == INPUT_ONLINE and len(self._rolling_fps._timestamps) >= min(
            10, self.fps_window_size
        )
        if fps_evaluated and current_fps < self.fps_minimum:
            reasons.append(f"Low FPS throughput ({current_fps:.1f} FPS < minimum {self.fps_minimum:.1f} FPS)")

        # 4. Evaluate Consecutive Pipeline Errors
        if self._metrics.consecutive_errors >= self.max_consecutive_errors_offline:
            reasons.append(f"Excessive consecutive pipeline errors ({self._metrics.consecutive_errors})")
        elif self._metrics.consecutive_errors >= self.max_consecutive_errors_degraded:
            reasons.append(f"Elevated consecutive pipeline errors ({self._metrics.consecutive_errors})")

        # 5. Evaluate Network & Remote Alert Delivery
        if self._network_status == NETWORK_UNAVAILABLE:
            reasons.append("Remote network / Telegram endpoint unavailable")
        if self._pending_alert_count > 0:
            reasons.append(f"Offline delivery queue has {self._pending_alert_count} pending alert(s)")

        # 6. Synthesize Overall System Health
        # OFFLINE: Fatal stream loss, failed detector, or fatal consecutive error cascade
        if (
            input_status == INPUT_OFFLINE
            or self._detector_status == SUBSYSTEM_FAILED
            or self._metrics.consecutive_errors >= self.max_consecutive_errors_offline
        ):
            overall_status = HEALTH_OFFLINE

        # DEGRADED: Partial issues (low FPS, frame drops, non-fatal subsystem degradation, delivery failure)
        elif (
            input_status == INPUT_DEGRADED
            or (fps_evaluated and current_fps < self.fps_minimum)
            or self._detector_status == SUBSYSTEM_DEGRADED
            or self._tracker_status in (SUBSYSTEM_DEGRADED, SUBSYSTEM_FAILED)
            or self._risk_engine_status in (SUBSYSTEM_DEGRADED, SUBSYSTEM_FAILED)
            or self._alert_delivery_status in (ALERT_DELIVERY_DEGRADED, ALERT_DELIVERY_FAILED)
            or self._pending_alert_count > 0
            or self._network_status == NETWORK_UNAVAILABLE
            or self._metrics.consecutive_errors >= self.max_consecutive_errors_degraded
            or self._metrics.consecutive_dropped_frames > 0
        ):
            overall_status = HEALTH_DEGRADED

        # UNKNOWN: Uninitialized or awaiting frames
        elif input_status == INPUT_UNKNOWN or (self._metrics.total_frames == 0 and self._camera_opened is None):
            overall_status = HEALTH_UNKNOWN
            if not reasons:
                reasons.append("Awaiting initial input frames")

        # HEALTHY: Everything nominal
        else:
            overall_status = HEALTH_HEALTHY
            if not reasons:
                reasons.append("All monitored subsystems operational")

        snapshot = HealthSnapshot(
            overall_status=overall_status,
            input_status=input_status,
            detector_status=self._detector_status,
            tracker_status=self._tracker_status,
            risk_engine_status=self._risk_engine_status,
            current_fps=current_fps,
            average_fps=self._metrics.average_fps(current_time),
            last_frame_age_seconds=self._metrics.last_frame_age(current_time),
            uptime_seconds=self._metrics.uptime(current_time),
            total_frames=self._metrics.total_frames,
            processed_frames=self._metrics.processed_frames,
            dropped_frames=self._metrics.dropped_frames,
            error_count=self._metrics.error_count,
            consecutive_errors=self._metrics.consecutive_errors,
            network_status=self._network_status,
            alert_delivery_status=self._alert_delivery_status,
            pending_alert_count=self._pending_alert_count,
            last_error=self._last_error_message,
            last_error_time=self._last_error_time,
            reasons=reasons,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time)),
        )

        return snapshot

    def get_snapshot(self, now: Optional[float] = None) -> HealthSnapshot:
        """Alias for evaluate() returning current health snapshot."""
        return self.evaluate(now=now)

    def check_transition(
        self,
        snapshot: HealthSnapshot,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Detects if a state transition occurred relative to the previous evaluation.

        Returns:
            (old_status, new_status, primary_reason) if changed, else None.
        """
        old_status = self._last_overall_status
        new_status = snapshot.overall_status

        if old_status is None:
            self._last_overall_status = new_status
            return None

        if old_status != new_status:
            self._last_overall_status = new_status
            primary_reason = snapshot.reasons[0] if snapshot.reasons else "State updated"
            return (old_status, new_status, primary_reason)

        return None
