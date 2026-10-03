"""
=============================================================================
🐘 PROJECT ZOGAN — TRACKER ALIAS MODULE
=============================================================================

This module provides the target `ai.tracker` interface, re-exporting all
classes, functions, and constants from `ai.tracking` for clean naming
and full backward/forward compatibility.

Usage:
    from ai.tracker import ElephantTracker, DIRECTION_RIGHT
=============================================================================
"""

from ai.tracking import (  # noqa: F401
    DEFAULT_MAX_HISTORY,
    DEFAULT_MAX_LOST_FRAMES,
    DEFAULT_MOVEMENT_THRESHOLD,
    DEFAULT_SMOOTHING_WINDOW,
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_STATIONARY,
    DIRECTION_UNKNOWN,
    DIRECTION_UP,
    VALID_DIRECTIONS,
    ElephantTracker,
    ElephantTrajectory,
    calculate_iou,
)

__all__ = [
    "DEFAULT_MAX_HISTORY",
    "DEFAULT_MAX_LOST_FRAMES",
    "DEFAULT_MOVEMENT_THRESHOLD",
    "DEFAULT_SMOOTHING_WINDOW",
    "DIRECTION_DOWN",
    "DIRECTION_LEFT",
    "DIRECTION_RIGHT",
    "DIRECTION_STATIONARY",
    "DIRECTION_UNKNOWN",
    "DIRECTION_UP",
    "VALID_DIRECTIONS",
    "ElephantTracker",
    "ElephantTrajectory",
    "calculate_iou",
]
