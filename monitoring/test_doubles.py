"""
=============================================================================
🐘 PROJECT ZOGAN — RELIABILITY TEST DOUBLES & FAILURE INJECTION (PHASE 8)
=============================================================================

This module provides deterministic test doubles and failure injectors
for testing monitoring pipeline reliability, fault tolerance, and recovery
without requiring physical cameras, GPUs, or unpredictable wall-clock timing.
=============================================================================
"""

import numpy as np
from typing import Any, List, Optional, Tuple


class FakeClock:
    """
    Deterministic simulated clock for precise timing and freshness tests.
    """

    def __init__(self, start_time: float = 1700000000.0):
        self._current_time: float = start_time

    def now(self) -> float:
        """Returns the current simulated timestamp."""
        return self._current_time

    def advance(self, seconds: float) -> float:
        """Advances simulated time by specified seconds."""
        self._current_time += seconds
        return self._current_time

    def set(self, timestamp: float) -> float:
        """Sets simulated time to an exact timestamp."""
        self._current_time = timestamp
        return self._current_time


class FakeCamera:
    """
    Test double for cv2.VideoCapture supporting controlled failure injection:
      - Simulating unopened camera
      - Simulating dropped frames
      - Simulating stream disconnection
      - Simulating frozen frames
    """

    def __init__(
        self,
        opened: bool = True,
        frame_shape: Tuple[int, int, int] = (480, 640, 3),
        fail_after_frames: Optional[int] = None,
        drop_every_n: Optional[int] = None,
        freeze_after_frames: Optional[int] = None,
    ):
        self._opened: bool = opened
        self._frame_shape: Tuple[int, int, int] = frame_shape
        self._fail_after_frames: Optional[int] = fail_after_frames
        self._drop_every_n: Optional[int] = drop_every_n
        self._freeze_after_frames: Optional[int] = freeze_after_frames
        self._frame_count: int = 0
        self._is_released: bool = False
        self._frozen_frame: Optional[np.ndarray] = None

    def isOpened(self) -> bool:
        """Returns True if the camera stream is considered opened."""
        return self._opened and not self._is_released

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Simulates frame read with failure injection rules.
        """
        if not self.isOpened():
            return False, None

        self._frame_count += 1

        # Check disconnect / fail after N frames
        if self._fail_after_frames is not None and self._frame_count > self._fail_after_frames:
            return False, None

        # Check periodic frame drop
        if self._drop_every_n is not None and (self._frame_count % self._drop_every_n == 0):
            return False, None

        # Check frozen frame
        if self._freeze_after_frames is not None and self._frame_count >= self._freeze_after_frames:
            if self._frozen_frame is None:
                self._frozen_frame = np.full(self._frame_shape, 128, dtype=np.uint8)
            return True, self._frozen_frame.copy()

        # Normal frame generation
        frame = np.zeros(self._frame_shape, dtype=np.uint8)
        # Add frame index stamp in top corner to make frame unique
        frame[0, 0, 0] = self._frame_count % 256
        return True, frame

    def release(self) -> None:
        """Releases the simulated camera stream."""
        self._is_released = True


class FaultyDetector:
    """
    Test double for the elephant detector allowing controlled inference failures.
    """

    def __init__(
        self,
        fail_on_load: bool = False,
        fail_on_predict: bool = False,
        fail_after_n_predictions: Optional[int] = None,
    ):
        self.fail_on_load: bool = fail_on_load
        self.fail_on_predict: bool = fail_on_predict
        self.fail_after_n_predictions: Optional[int] = fail_after_n_predictions
        self.prediction_count: int = 0
        self.is_loaded: bool = not fail_on_load

    def load(self, *args, **kwargs) -> None:
        if self.fail_on_load:
            raise RuntimeError("Injected detector initialization failure")
        self.is_loaded = True

    def predict(self, *args, **kwargs) -> List[Any]:
        self.prediction_count += 1
        if self.fail_on_predict:
            raise RuntimeError("Injected detector inference failure")
        if self.fail_after_n_predictions is not None and self.prediction_count > self.fail_after_n_predictions:
            raise RuntimeError("Injected detector crash after N inferences")
        return []


class FaultyTracker:
    """
    Test double for the multi-object tracker allowing controlled tracking failures.
    """

    def __init__(
        self,
        fail_on_update: bool = False,
        fail_after_n_updates: Optional[int] = None,
    ):
        self.fail_on_update: bool = fail_on_update
        self.fail_after_n_updates: Optional[int] = fail_after_n_updates
        self.update_count: int = 0

    def update(self, *args, **kwargs) -> List[Any]:
        self.update_count += 1
        if self.fail_on_update:
            raise RuntimeError("Injected tracking engine failure")
        if self.fail_after_n_updates is not None and self.update_count > self.fail_after_n_updates:
            raise RuntimeError("Injected tracking crash after N frames")
        return []


class FaultyRiskEngine:
    """
    Test double for the risk assessment engine allowing controlled risk evaluation failures.
    """

    def __init__(self, fail_on_evaluate: bool = False):
        self.fail_on_evaluate: bool = fail_on_evaluate

    def evaluate(self, *args, **kwargs) -> Any:
        if self.fail_on_evaluate:
            raise RuntimeError("Injected risk engine calculation failure")
        return None
