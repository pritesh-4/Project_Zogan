"""
=============================================================================
🐘 PROJECT ZOGAN — ALERT EVENT DATA MODEL (PHASE 6)
=============================================================================

This module defines the structured AlertEvent representation for confirmed
elephant detection incidents.

Each event includes:
  - event ID (unique incident identifier)
  - timestamp (ISO-8601 formatted timestamp)
  - risk_level (LOW, MEDIUM, HIGH, CRITICAL)
  - risk_score (0–100 integer score from Risk Engine)
  - detection confidence (float, e.g. 0.94)
  - track ID (temporary ByteTrack tracking ID)
  - group size (count of simultaneously detected elephants)
  - zone (e.g. WARNING, BUFFER, VILLAGE, FOREST)
  - distance to sensitive zone (meters)
  - movement trend (APPROACHING, RECEDING, STABLE, or image-space)
  - simulation mode status (boolean)

⚠️ Do NOT fabricate values when data is unavailable.
=============================================================================
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


class AlertEvent:
    """
    Structured alert event representing a confirmed elephant detection incident.
    """

    def __init__(
        self,
        event_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        risk_level: Optional[str] = None,
        risk_score: Optional[int] = None,
        confidence: Optional[float] = None,
        track_id: Optional[int] = None,
        group_size: Optional[int] = None,
        zone: Optional[str] = None,
        distance_m: Optional[Union[float, int]] = None,
        movement: Optional[str] = None,
        simulation: Optional[bool] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.event_id: str = event_id or f"evt_{uuid.uuid4().hex[:12]}"
        self.timestamp: str = timestamp or datetime.now(timezone.utc).isoformat()
        self.risk_level: Optional[str] = str(risk_level).upper() if risk_level is not None else None
        self.risk_score: Optional[int] = int(risk_score) if risk_score is not None else None
        self.confidence: Optional[float] = float(confidence) if confidence is not None else None
        self.track_id: Optional[int] = int(track_id) if track_id is not None else None
        self.group_size: Optional[int] = int(group_size) if group_size is not None else None
        self.zone: Optional[str] = str(zone).upper() if zone is not None else None
        self.distance_m: Optional[Union[float, int]] = round(float(distance_m), 1) if distance_m is not None else None
        self.movement: Optional[str] = str(movement).upper() if movement is not None else None
        self.simulation: Optional[bool] = bool(simulation) if simulation is not None else None
        self.extra: Dict[str, Any] = extra or {}

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """
        Serializes the event into a dictionary representation.
        When exclude_none is True, keys whose values are None are excluded
        so non-existent data is never fabricated.
        """
        raw_dict = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "track_id": self.track_id,
            "group_size": self.group_size,
            "zone": self.zone,
            "distance_m": self.distance_m,
            "movement": self.movement,
            "simulation": self.simulation,
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
            risk_level=data.get("risk_level") or data.get("alert_level"),
            risk_score=data.get("risk_score"),
            confidence=data.get("confidence"),
            track_id=data.get("track_id"),
            group_size=data.get("group_size"),
            zone=data.get("zone"),
            distance_m=data.get("distance_m"),
            movement=data.get("movement"),
            simulation=data.get("simulation"),
        )

    def __repr__(self) -> str:
        return (
            f"AlertEvent(id='{self.event_id}', risk='{self.risk_level}', "
            f"score={self.risk_score}, conf={self.confidence}, track=#{self.track_id})"
        )


def create_alert_event(
    confidence: Optional[float] = None,
    tracked_info: Optional[List[Dict[str, Any]]] = None,
    risk_info: Optional[Dict[str, Any]] = None,
    simulation: Optional[bool] = True,
    event_id: Optional[str] = None,
    timestamp: Optional[str] = None,
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
    risk_level = None
    risk_score = None
    zone = None
    distance_m = None

    if risk_info:
        risk_level = risk_info.get("risk_level")
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
            simulation = risk_info.get("geo_mode") == "SIMULATION"

    if risk_level is None and confidence is not None:
        # Default alert level if risk engine was not evaluated
        risk_level = "HIGH"

    return AlertEvent(
        event_id=event_id,
        timestamp=timestamp,
        risk_level=risk_level,
        risk_score=risk_score,
        confidence=confidence,
        track_id=track_id,
        group_size=group_size,
        zone=zone,
        distance_m=distance_m,
        movement=movement,
        simulation=simulation,
    )
