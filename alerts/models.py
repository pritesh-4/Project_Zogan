"""
=============================================================================
🐘 PROJECT ZOGAN — ALERT EVENT DATA MODEL (PHASE 6)
=============================================================================

This module defines the structured AlertEvent representation for confirmed
elephant detection incidents.

Each event includes:
  - event_id (unique incident identifier, e.g. 'zogan-8c9f...')
  - timestamp (ISO-8601 formatted timestamp)
  - alert_level / risk_level (LOW, MEDIUM, HIGH, CRITICAL)
  - risk_score (0–100 integer score from Risk Engine)
  - detection confidence (float, e.g. 0.94)
  - track_id (temporary ByteTrack tracking ID)
  - group_size (count of simultaneously detected elephants)
  - zone (e.g. WARNING, BUFFER, VILLAGE, FOREST)
  - distance_m (meters to sensitive zone)
  - movement (trend: APPROACHING, RECEDING, STABLE, or image-space)
  - simulation_mode (boolean)
  - camera_id (optional camera identifier, e.g. 'cam-01')

⚠️ Do NOT fabricate values when data is unavailable.
=============================================================================
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

# Delivery Status Constants (Phase 9)
DELIVERY_PENDING: str = "PENDING"
DELIVERY_SENT: str = "SENT"
DELIVERY_FAILED: str = "FAILED"
DELIVERY_RETRYING: str = "RETRYING"
DELIVERY_NOT_CONFIGURED: str = "NOT_CONFIGURED"
DELIVERY_SKIPPED: str = "SKIPPED"


class AlertEvent:
    """
    Structured alert event representing a confirmed elephant detection incident.
    """

    def __init__(
        self,
        event_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        alert_level: Optional[str] = None,
        risk_level: Optional[str] = None,
        risk_score: Optional[int] = None,
        confidence: Optional[float] = None,
        track_id: Optional[int] = None,
        group_size: Optional[int] = None,
        zone: Optional[str] = None,
        distance_m: Optional[Union[float, int]] = None,
        movement: Optional[str] = None,
        simulation_mode: Optional[bool] = None,
        simulation: Optional[bool] = None,
        camera_id: Optional[str] = None,
        reasons: Optional[List[str]] = None,
        contributing_factors: Optional[Dict[str, int]] = None,
        detection_count: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        event_state: Optional[str] = None,
        is_resolved: Optional[bool] = None,
        resolution_time: Optional[str] = None,
        delivery_status: Optional[str] = None,
        delivery_attempts: Optional[int] = None,
        last_delivery_attempt: Optional[str] = None,
        delivery_error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.event_id: str = event_id or f"zogan-{uuid.uuid4().hex[:12]}"
        self.timestamp: str = timestamp or datetime.now(timezone.utc).isoformat()

        # Support both alert_level and risk_level parameters
        level = alert_level if alert_level is not None else risk_level
        self._alert_level: Optional[str] = str(level).upper() if level is not None else None

        self.risk_score: Optional[int] = int(risk_score) if risk_score is not None else None
        self.confidence: Optional[float] = float(confidence) if confidence is not None else None
        self.track_id: Optional[int] = int(track_id) if track_id is not None else None
        self.group_size: Optional[int] = int(group_size) if group_size is not None else None
        self.zone: Optional[str] = str(zone).upper() if zone is not None else None
        self.distance_m: Optional[Union[float, int]] = round(float(distance_m), 1) if distance_m is not None else None
        self.movement: Optional[str] = str(movement).upper() if movement is not None else None

        # Support both simulation_mode and simulation parameters
        sim = simulation_mode if simulation_mode is not None else simulation
        self._simulation_mode: Optional[bool] = bool(sim) if sim is not None else None

        self.camera_id: Optional[str] = str(camera_id) if camera_id is not None else None
        self.reasons: Optional[List[str]] = list(reasons) if reasons is not None else None
        self.contributing_factors: Optional[Dict[str, int]] = (
            dict(contributing_factors) if contributing_factors is not None else None
        )
        self.detection_count: Optional[int] = int(detection_count) if detection_count is not None else None
        self.duration_seconds: Optional[float] = float(duration_seconds) if duration_seconds is not None else None
        self.event_state: Optional[str] = str(event_state).upper() if event_state is not None else None
        self.is_resolved: Optional[bool] = bool(is_resolved) if is_resolved is not None else None
        self.resolution_time: Optional[str] = str(resolution_time) if resolution_time is not None else None
        self.delivery_status: Optional[str] = str(delivery_status).upper() if delivery_status is not None else None
        self.delivery_attempts: int = int(delivery_attempts) if delivery_attempts is not None else 0
        self.last_delivery_attempt: Optional[str] = (
            str(last_delivery_attempt) if last_delivery_attempt is not None else None
        )
        self.delivery_error: Optional[str] = str(delivery_error) if delivery_error is not None else None
        self.extra: Dict[str, Any] = extra or {}

    @property
    def alert_level(self) -> Optional[str]:
        return self._alert_level

    @alert_level.setter
    def alert_level(self, value: Optional[str]) -> None:
        self._alert_level = str(value).upper() if value is not None else None

    @property
    def risk_level(self) -> Optional[str]:
        return self._alert_level

    @risk_level.setter
    def risk_level(self, value: Optional[str]) -> None:
        self._alert_level = str(value).upper() if value is not None else None

    @property
    def simulation_mode(self) -> Optional[bool]:
        return self._simulation_mode

    @simulation_mode.setter
    def simulation_mode(self, value: Optional[bool]) -> None:
        self._simulation_mode = bool(value) if value is not None else None

    @property
    def simulation(self) -> Optional[bool]:
        return self._simulation_mode

    @simulation.setter
    def simulation(self, value: Optional[bool]) -> None:
        self._simulation_mode = bool(value) if value is not None else None

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """
        Serializes the event into a dictionary representation.
        When exclude_none is True, keys whose values are None are excluded
        so non-existent data is never fabricated.
        """
        raw_dict = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "alert_level": self._alert_level,
            "risk_level": self._alert_level,
            "risk_score": self.risk_score,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "track_id": self.track_id,
            "group_size": self.group_size,
            "zone": self.zone,
            "distance_m": self.distance_m,
            "movement": self.movement,
            "simulation_mode": self._simulation_mode,
            "simulation": self._simulation_mode,
            "camera_id": self.camera_id,
            "reasons": self.reasons,
            "contributing_factors": self.contributing_factors,
            "detection_count": self.detection_count,
            "duration_seconds": round(self.duration_seconds, 2) if self.duration_seconds is not None else None,
            "event_state": self.event_state,
            "is_resolved": self.is_resolved,
            "resolution_time": self.resolution_time,
            "delivery_status": self.delivery_status,
            "delivery_attempts": self.delivery_attempts if self.delivery_attempts > 0 else None,
            "last_delivery_attempt": self.last_delivery_attempt,
            "delivery_error": self.delivery_error,
        }

        if exclude_none:
            return {k: v for k, v in raw_dict.items() if v is not None}
        return raw_dict

    def to_json(self) -> str:
        """
        Serializes existing event attributes to a JSON string.
        """
        import json

        return json.dumps(self.to_dict(exclude_none=True), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertEvent":
        """
        Constructs an AlertEvent from a dictionary.
        """
        return cls(
            event_id=data.get("event_id"),
            timestamp=data.get("timestamp"),
            alert_level=data.get("alert_level") or data.get("risk_level"),
            risk_score=data.get("risk_score"),
            confidence=data.get("confidence"),
            track_id=data.get("track_id"),
            group_size=data.get("group_size"),
            zone=data.get("zone"),
            distance_m=data.get("distance_m"),
            movement=data.get("movement"),
            simulation_mode=data.get("simulation_mode") if "simulation_mode" in data else data.get("simulation"),
            camera_id=data.get("camera_id"),
            reasons=data.get("reasons"),
            contributing_factors=data.get("contributing_factors") or data.get("breakdown"),
            detection_count=data.get("detection_count"),
            duration_seconds=data.get("duration_seconds"),
            event_state=data.get("event_state"),
            is_resolved=data.get("is_resolved"),
            resolution_time=data.get("resolution_time"),
            delivery_status=data.get("delivery_status"),
            delivery_attempts=data.get("delivery_attempts", 0),
            last_delivery_attempt=data.get("last_delivery_attempt"),
            delivery_error=data.get("delivery_error"),
        )

    def __repr__(self) -> str:
        return (
            f"AlertEvent(id='{self.event_id}', level='{self.alert_level}', "
            f"score={self.risk_score}, conf={self.confidence}, track=#{self.track_id})"
        )


def create_alert_event(
    confidence: Optional[float] = None,
    tracked_info: Optional[List[Dict[str, Any]]] = None,
    risk_info: Optional[Dict[str, Any]] = None,
    simulation_mode: Optional[bool] = None,
    simulation: Optional[bool] = None,
    event_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    camera_id: Optional[str] = "cam-01",
    reasons: Optional[List[str]] = None,
    contributing_factors: Optional[Dict[str, int]] = None,
    detection_count: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    event_state: Optional[str] = None,
    is_resolved: Optional[bool] = None,
    resolution_time: Optional[str] = None,
    delivery_status: Optional[str] = None,
    delivery_attempts: Optional[int] = None,
    last_delivery_attempt: Optional[str] = None,
    delivery_error: Optional[str] = None,
) -> AlertEvent:
    """
    Factory function to construct an AlertEvent from detection, tracking,
    and risk assessment pipeline inputs.
    """
    # Track ID & initial movement from tracked_info
    track_id = None
    movement = None
    group_size = None

    if tracked_info and len(tracked_info) > 0:
        first_track = tracked_info[0]
        track_id = first_track.get("id", first_track.get("track_id"))
        movement = first_track.get("movement")
        group_size = len(tracked_info)

    # Enrich from risk_info if provided
    alert_level = None
    risk_score = None
    zone = None
    distance_m = None

    sim = simulation_mode if simulation_mode is not None else simulation
    if sim is None:
        sim = True

    ev_id = event_id

    if risk_info:
        alert_level = risk_info.get("risk_level") or risk_info.get("alert_level")
        risk_score = risk_info.get("risk_score")
        zone = risk_info.get("zone")
        distance_m = risk_info.get("distance_to_protected_m")
        # Prefer geographic approach trend if available
        trend = risk_info.get("trend")
        if trend and trend != "UNKNOWN":
            movement = trend
        if risk_info.get("group_size") is not None:
            group_size = risk_info.get("group_size")
        if "geo_mode" in risk_info:
            sim = risk_info.get("geo_mode") == "SIMULATION"
        if not ev_id and risk_info.get("event_id"):
            ev_id = risk_info.get("event_id")
        if reasons is None and risk_info.get("reasons"):
            reasons = risk_info.get("reasons")
        if contributing_factors is None and (risk_info.get("contributing_factors") or risk_info.get("breakdown")):
            contributing_factors = risk_info.get("contributing_factors") or risk_info.get("breakdown")
        if detection_count is None and risk_info.get("detection_count") is not None:
            detection_count = risk_info.get("detection_count")
        if duration_seconds is None and risk_info.get("duration_seconds") is not None:
            duration_seconds = risk_info.get("duration_seconds")
        if event_state is None and risk_info.get("event_state") is not None:
            event_state = risk_info.get("event_state")
        if is_resolved is None and risk_info.get("is_resolved") is not None:
            is_resolved = risk_info.get("is_resolved")
        if resolution_time is None and risk_info.get("resolution_time") is not None:
            resolution_time = risk_info.get("resolution_time")

    if alert_level is None and confidence is not None:
        # Default alert level if risk engine was not evaluated
        alert_level = "HIGH"

    return AlertEvent(
        event_id=ev_id,
        timestamp=timestamp,
        alert_level=alert_level,
        risk_score=risk_score,
        confidence=confidence,
        track_id=track_id,
        group_size=group_size,
        zone=zone,
        distance_m=distance_m,
        movement=movement,
        simulation_mode=sim,
        camera_id=camera_id,
        reasons=reasons,
        contributing_factors=contributing_factors,
        detection_count=detection_count,
        duration_seconds=duration_seconds,
        event_state=event_state,
        is_resolved=is_resolved,
        resolution_time=resolution_time,
        delivery_status=delivery_status,
        delivery_attempts=delivery_attempts,
        last_delivery_attempt=last_delivery_attempt,
        delivery_error=delivery_error,
    )
