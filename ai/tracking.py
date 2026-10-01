"""
=============================================================================
🐘 PROJECT ZOGAN — OBJECT TRACKING & MOVEMENT ANALYSIS (PHASE 4)
=============================================================================

This module provides object tracking state management and image-space
movement estimation for detected elephants across consecutive video frames.

Pipeline Position:
  CAMERA -> CUSTOM YOLO MODEL -> ELEPHANT DETECTION -> TRACKER -> TRACK ID -> POSITION HISTORY -> MOVEMENT ESTIMATION

Key Features:
  - Temporary session Track ID assignment
  - Bounding-box center calculation: center = ((x1 + x2)/2, (y1 + y2)/2)
  - Bounded position history (default MAX_HISTORY = 20)
  - Image-space movement direction estimation:
      RIGHT, LEFT, UP, DOWN, STATIONARY, UNKNOWN
  - Configurable pixel movement threshold (default: 10 pixels) to avoid jitter
  - Multi-frame window smoothing to suppress detection noise
  - Multiple simultaneous elephant track maintenance
  - Lost-track tolerance and automatic expiration cleanup

⚠️ IMPORTANT PHASE 4 LIMITATIONS & CONSTRAINTS:
  1. Direction is IMAGE-SPACE ONLY (not geographic / compass direction like North/South/East/West).
  2. The CAMERA IS ASSUMED STATIONARY (no ego-motion or camera movement compensation yet).
  3. Tracking IDs are TEMPORARY SESSION IDs (not permanent biological IDs of real elephants).
  4. Movement speed is pixel-based, not physical real-world metric speed.
  5. Pixel movement does not equal physical distance.
  6. No geographic location / GPS or geofencing is known yet.
  7. No danger score or risk prediction is attached to movement yet.
  8. The system does not predict elephant behavior or intent.
=============================================================================
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

# =============================================================================
# ⚙️ CONFIGURATION CONSTANTS & DEFAULTS
# =============================================================================

DEFAULT_MAX_HISTORY: int = 20
DEFAULT_MOVEMENT_THRESHOLD: float = 10.0  # Pixels
DEFAULT_MAX_LOST_FRAMES: int = 30  # ~1.0s at 30 fps
DEFAULT_SMOOTHING_WINDOW: int = 5  # Frames evaluated for stable trend

# Image-Space Direction Labels
DIRECTION_RIGHT: str = "RIGHT"
DIRECTION_LEFT: str = "LEFT"
DIRECTION_UP: str = "UP"
DIRECTION_DOWN: str = "DOWN"
DIRECTION_STATIONARY: str = "STATIONARY"
DIRECTION_UNKNOWN: str = "UNKNOWN"

VALID_DIRECTIONS = {
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
    DIRECTION_UP,
    DIRECTION_DOWN,
    DIRECTION_STATIONARY,
    DIRECTION_UNKNOWN,
}


# =============================================================================
# 📐 GEOMETRY & DIRECTION ESTIMATION FUNCTIONS
# =============================================================================


def calculate_center(bbox: Union[Tuple, List]) -> Tuple[float, float]:
    """
    Calculates the center point (center_x, center_y) of a bounding box.
    Box format: (x1, y1, x2, y2)
    """
    if len(bbox) < 4:
        raise ValueError(f"Expected bbox with at least 4 coordinates (x1, y1, x2, y2), got: {bbox}")
    x1, y1, x2, y2 = bbox[:4]
    center_x = (float(x1) + float(x2)) / 2.0
    center_y = (float(y1) + float(y2)) / 2.0
    return (center_x, center_y)


def estimate_direction(
    prev_point: Optional[Tuple[float, float]],
    curr_point: Optional[Tuple[float, float]],
    threshold: float = DEFAULT_MOVEMENT_THRESHOLD,
) -> str:
    """
    Estimates 2D image-space movement direction between two points.

    In image space coordinates:
      - x increases to the RIGHT (dx > 0)
      - x decreases to the LEFT  (dx < 0)
      - y increases DOWNWARDS    (dy > 0)
      - y decreases UPWARDS      (dy < 0)

    If Euclidean distance is strictly less than threshold:
      Returns STATIONARY (filters detector noise and bounding-box jitter).
    Otherwise:
      Compares dominant axis (|dx| vs |dy|):
        - If |dx| >= |dy|: RIGHT (dx > 0) or LEFT (dx < 0)
        - If |dy| > |dx|:  DOWN (dy > 0) or UP (dy < 0)
    """
    if prev_point is None or curr_point is None:
        return DIRECTION_UNKNOWN

    dx = curr_point[0] - prev_point[0]
    dy = curr_point[1] - prev_point[1]
    distance = math.hypot(dx, dy)

    if distance < threshold:
        return DIRECTION_STATIONARY

    if abs(dx) >= abs(dy):
        return DIRECTION_RIGHT if dx > 0 else DIRECTION_LEFT
    else:
        return DIRECTION_DOWN if dy > 0 else DIRECTION_UP


# =============================================================================
# 🐘 TRACKED ELEPHANT ENTITY
# =============================================================================


class TrackedElephant:
    """
    Maintains tracking state and position history for a single tracked elephant.
    """

    def __init__(
        self,
        track_id: int,
        initial_bbox: Union[Tuple, List],
        confidence: float = 0.0,
        frame_idx: int = 0,
        max_history: int = DEFAULT_MAX_HISTORY,
        movement_threshold: float = DEFAULT_MOVEMENT_THRESHOLD,
        window_size: int = DEFAULT_SMOOTHING_WINDOW,
    ):
        self.track_id: int = int(track_id)
        self.max_history: int = max_history
        self.movement_threshold: float = float(movement_threshold)
        self.window_size: int = max(2, window_size)

        self.history: deque = deque(maxlen=max_history)
        self.bbox: Tuple[float, float, float, float] = tuple(initial_bbox[:4])
        self.confidence: float = float(confidence)
        self.first_seen_frame: int = frame_idx
        self.last_seen_frame: int = frame_idx
        self.frames_lost: int = 0

        # Calculate and record initial center position
        center = calculate_center(initial_bbox)
        self.history.append(center)

    def update(
        self,
        bbox: Union[Tuple, List],
        confidence: float,
        frame_idx: int,
    ) -> None:
        """
        Updates the track with a new detection observation in the current frame.
        """
        self.bbox = tuple(bbox[:4])
        self.confidence = float(confidence)
        self.last_seen_frame = frame_idx
        self.frames_lost = 0

        center = calculate_center(bbox)
        self.history.append(center)

    def mark_missed(self) -> None:
        """
        Increments the consecutive missed frame count when not detected in a frame.
        """
        self.frames_lost += 1

    @property
    def current_center(self) -> Optional[Tuple[float, float]]:
        """Returns the most recently recorded center point."""
        return self.history[-1] if self.history else None

    @property
    def previous_center(self) -> Optional[Tuple[float, float]]:
        """Returns the immediately preceding center point (if available)."""
        return self.history[-2] if len(self.history) >= 2 else None

    @property
    def movement(self) -> str:
        """
        Returns estimated image-space direction using the configured smoothing window.
        """
        return self.get_movement()

    def get_movement(
        self,
        threshold: Optional[float] = None,
        window: Optional[int] = None,
    ) -> str:
        """
        Estimates movement direction across a specified window length.
        Comparing against an anchor 'window' frames ago suppresses jitter.
        """
        if len(self.history) < 2:
            return DIRECTION_UNKNOWN

        thresh = self.movement_threshold if threshold is None else threshold
        win = self.window_size if window is None else max(2, window)

        # Reference point from 'win' frames ago, or oldest in history
        ref_idx = max(0, len(self.history) - win)
        ref_point = self.history[ref_idx]
        curr_point = self.history[-1]

        return estimate_direction(ref_point, curr_point, threshold=thresh)

    def get_displacement(self, window: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Calculates (dx, dy, distance) in pixels across the evaluation window.
        """
        if len(self.history) < 2:
            return (0.0, 0.0, 0.0)

        win = self.window_size if window is None else max(2, window)
        ref_idx = max(0, len(self.history) - win)
        ref_point = self.history[ref_idx]
        curr_point = self.history[-1]

        dx = curr_point[0] - ref_point[0]
        dy = curr_point[1] - ref_point[1]
        dist = math.hypot(dx, dy)
        return (dx, dy, dist)

    def to_dict(self) -> dict:
        """Returns a serializable dictionary representation of the track."""
        dx, dy, dist = self.get_displacement()
        return {
            "track_id": self.track_id,
            "center": self.current_center,
            "previous_center": self.previous_center,
            "history": list(self.history),
            "bbox": self.bbox,
            "confidence": self.confidence,
            "movement": self.movement,
            "displacement_px": round(dist, 2),
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "frames_lost": self.frames_lost,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
        }


# =============================================================================
# 🎯 HIGH-LEVEL ELEPHANT TRACKER MANAGER
# =============================================================================


class ElephantTracker:
    """
    High-level manager for multiple tracked elephants across video frames.
    Maintains active tracks, updates histories, computes movement,
    and cleans up expired tracks.
    """

    def __init__(
        self,
        max_history: int = DEFAULT_MAX_HISTORY,
        movement_threshold: float = DEFAULT_MOVEMENT_THRESHOLD,
        max_lost_frames: int = DEFAULT_MAX_LOST_FRAMES,
        window_size: int = DEFAULT_SMOOTHING_WINDOW,
    ):
        self.max_history: int = max_history
        self.movement_threshold: float = float(movement_threshold)
        self.max_lost_frames: int = max_lost_frames
        self.window_size: int = window_size

        self.tracks: Dict[int, TrackedElephant] = {}
        self.frame_count: int = 0

    def update(
        self,
        detections: List[Union[dict, tuple, list]],
        frame_idx: Optional[int] = None,
    ) -> List[TrackedElephant]:
        """
        Updates the tracker with detections observed in the current frame.

        Supported input formats for detections:
          1. List of dicts:
             [{'id': 1, 'bbox': (x1, y1, x2, y2), 'conf': 0.95}, ...]
          2. List of tuples/lists:
             [(track_id, (x1, y1, x2, y2), conf), ...]

        Returns list of active TrackedElephant objects present in this frame.
        """
        if frame_idx is None:
            self.frame_count += 1
            frame_idx = self.frame_count
        else:
            self.frame_count = frame_idx

        observed_ids = set()
        active_in_frame: List[TrackedElephant] = []

        for det in detections:
            if isinstance(det, dict):
                tid = int(det.get("id", det.get("track_id", 0)))
                bbox = det.get("bbox", det.get("box", (0, 0, 0, 0)))
                conf = float(det.get("conf", det.get("confidence", 1.0)))
            elif isinstance(det, (tuple, list)):
                if len(det) < 2:
                    continue
                tid = int(det[0])
                bbox = det[1]
                conf = float(det[2]) if len(det) > 2 else 1.0
            else:
                continue

            observed_ids.add(tid)

            if tid in self.tracks:
                # Update existing track
                track = self.tracks[tid]
                track.update(bbox, conf, frame_idx)
            else:
                # Initialize new track
                track = TrackedElephant(
                    track_id=tid,
                    initial_bbox=bbox,
                    confidence=conf,
                    frame_idx=frame_idx,
                    max_history=self.max_history,
                    movement_threshold=self.movement_threshold,
                    window_size=self.window_size,
                )
                self.tracks[tid] = track

            active_in_frame.append(track)

        # Mark missed frames for existing tracks not present in this frame
        dead_ids = []
        for tid, track in self.tracks.items():
            if tid not in observed_ids:
                track.mark_missed()
                if track.frames_lost > self.max_lost_frames:
                    dead_ids.append(tid)

        # Evict expired dead tracks
        for tid in dead_ids:
            del self.tracks[tid]

        return active_in_frame

    def get_track(self, track_id: int) -> Optional[TrackedElephant]:
        """Retrieves a track by its integer track ID."""
        return self.tracks.get(int(track_id))

    def get_active_tracks(self) -> List[TrackedElephant]:
        """
        Returns all tracks observed in the most recent frame (frames_lost == 0).
        """
        return [t for t in self.tracks.values() if t.frames_lost == 0]

    def get_all_tracks(self) -> List[TrackedElephant]:
        """
        Returns all alive tracks in memory (including temporarily lost tracks).
        """
        return list(self.tracks.values())

    def get_hud_summary(self, max_items: int = 3) -> str:
        """
        Generates a concise string summary of active tracks for HUD display.
        Example: "#1 -> RIGHT  |  #2 -> STATIONARY"
        """
        active = self.get_active_tracks()
        if not active:
            return ""

        # Sort by track ID for consistent display
        active_sorted = sorted(active, key=lambda t: t.track_id)
        items = [f"#{t.track_id} -> {t.movement}" for t in active_sorted[:max_items]]

        summary = "  |  ".join(items)
        if len(active_sorted) > max_items:
            summary += f" (+{len(active_sorted) - max_items} more)"
        return summary

    def clear(self) -> None:
        """Resets all tracked objects and counters."""
        self.tracks.clear()
        self.frame_count = 0
