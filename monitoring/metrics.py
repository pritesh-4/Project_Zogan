"""
=============================================================================
🐘 PROJECT ZOGAN — RUNTIME METRICS & FPS MONITORING (PHASE 8)
=============================================================================

This module provides lightweight, deterministic performance and observability
metrics for the Zogan monitoring pipeline:
  - Rolling window FPS calculation (no single-frame jitter)
  - Frame throughput & dropped frame tracking
  - Bounded memory error tracking
  - Uptime and frame latency calculations
=============================================================================
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


class RollingFPS:
    """
    Computes runtime frames-per-second using a bounded sliding window
    of frame arrival timestamps.

    Avoids single-frame jitter while providing responsive detection
    of pipeline stalls and throughput drops.
    """

    def __init__(self, window_size: int = 30):
        if window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {window_size}")
        self._window_size = window_size
        self._timestamps: deque = deque(maxlen=window_size)
        self._min_observed_fps: float = float("inf")
        self._max_observed_fps: float = 0.0
        self._total_samples: int = 0

    def update(self, now: Optional[float] = None) -> float:
        """
        Records a frame timestamp and computes current rolling FPS.
        """
        current_time = time.time() if now is None else now
        self._timestamps.append(current_time)
        self._total_samples += 1

        fps = self.current_fps
        # Only record min/max once the window has accumulated meaningful samples
        if len(self._timestamps) >= min(5, self._window_size) and fps > 0:
            if fps < self._min_observed_fps:
                self._min_observed_fps = fps
            if fps > self._max_observed_fps:
                self._max_observed_fps = fps

        return fps

    @property
    def current_fps(self) -> float:
        """Calculates current FPS based on the sliding window timestamps."""
        count = len(self._timestamps)
        if count < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (count - 1) / span

    @property
    def min_fps(self) -> float:
        """Returns the minimum observed FPS (0.0 if not yet established)."""
        return 0.0 if self._min_observed_fps == float("inf") else self._min_observed_fps

    @property
    def max_fps(self) -> float:
        """Returns the maximum observed FPS."""
        return self._max_observed_fps

    @property
    def window_size(self) -> int:
        return self._window_size

    def reset(self) -> None:
        """Resets the sliding window and min/max statistics."""
        self._timestamps.clear()
        self._min_observed_fps = float("inf")
        self._max_observed_fps = 0.0
        self._total_samples = 0


@dataclass
class RuntimeMetrics:
    """
    Lightweight, observable runtime counters and timing metadata.
    """

    start_time: float
    total_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    consecutive_dropped_frames: int = 0
    detection_count: int = 0
    active_tracks: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_heartbeat_time: float = 0.0
    last_frame_time: Optional[float] = None
    last_processed_time: Optional[float] = None

    def uptime(self, now: Optional[float] = None) -> float:
        """Returns elapsed uptime in seconds."""
        current_time = time.time() if now is None else now
        return max(0.0, current_time - self.start_time)

    def last_frame_age(self, now: Optional[float] = None) -> float:
        """
        Returns age in seconds since the last successful frame arrived.
        Returns infinity if no frame has ever arrived.
        """
        if self.last_frame_time is None:
            return float("inf")
        current_time = time.time() if now is None else now
        return max(0.0, current_time - self.last_frame_time)

    def average_fps(self, now: Optional[float] = None) -> float:
        """Calculates total processed frames divided by total uptime."""
        up = self.uptime(now)
        if up <= 0:
            return 0.0
        return self.processed_frames / up
