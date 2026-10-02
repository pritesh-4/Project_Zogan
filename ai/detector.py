"""
=============================================================================
🐘 PROJECT ZOGAN — DETECTOR ABSTRACTION
=============================================================================

Clean model interface that encapsulates YOLO model loading, validation,
and inference. Designed so the model can be replaced without rewriting
camera code, tracker, risk engine, or alert engine.

Usage:
    detector = Detector()
    detector.load("models/elephant_v1/best.pt")
    detections = detector.predict(frame, conf_threshold=0.70)
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection result."""

    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    track_id: Optional[int] = None


@dataclass
class DetectorMetadata:
    """Information about the loaded model."""

    model_path: str = ""
    model_label: str = ""
    class_names: Dict[int, str] = field(default_factory=dict)
    target_class_id: Optional[int] = None
    target_class_found: bool = False


class Detector:
    """
    Encapsulates YOLO model loading, validation, and inference.

    Provides a clean interface:
      - load(path) → loads and validates model
      - predict(frame) → returns list of Detection objects
      - metadata() → returns model metadata
    """

    def __init__(self, target_class: str = "elephant"):
        self._model: Optional[YOLO] = None
        self._target_class = target_class.lower()
        self._metadata = DetectorMetadata()
        self._tracker_config: Optional[str] = None

    def load(
        self,
        model_path: str,
        model_label: str = "",
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        """
        Loads a YOLO model from the specified path and validates
        that the target class exists in the model's class map.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If target class not found in model.
            RuntimeError: If model fails to load.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        try:
            self._model = YOLO(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{model_path}': {e}") from e

        # Validate target class
        class_names = self._model.names
        target_id = None
        for cid, cname in class_names.items():
            if cname.lower() == self._target_class:
                target_id = cid
                break

        if target_id is None:
            raise ValueError(f"Model does not contain '{self._target_class}' class. Available: {class_names}")

        self._tracker_config = tracker_config
        self._metadata = DetectorMetadata(
            model_path=str(path),
            model_label=model_label or path.stem,
            class_names=dict(class_names),
            target_class_id=target_id,
            target_class_found=True,
        )

        logger.info(
            "Model loaded: %s (target=%s, class_id=%d, classes=%d)",
            model_path,
            self._target_class,
            target_id,
            len(class_names),
        )

    @property
    def is_loaded(self) -> bool:
        """Returns True if a model is loaded and validated."""
        return self._model is not None and self._metadata.target_class_found

    def metadata(self) -> DetectorMetadata:
        """Returns metadata about the loaded model."""
        return self._metadata

    def predict(
        self,
        frame,
        conf_threshold: float = 0.70,
        use_tracking: bool = True,
    ) -> List[Detection]:
        """
        Runs inference on a single frame and returns all detections.

        Args:
            frame: BGR numpy array from cv2
            conf_threshold: Minimum confidence to include
            use_tracking: If True, uses ByteTrack for persistent IDs

        Returns:
            List of Detection objects for ALL detected classes
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load() first.")

        if use_tracking and self._tracker_config:
            results = self._model.track(frame, persist=True, tracker=self._tracker_config, verbose=False)
        else:
            results = self._model.predict(source=frame, conf=conf_threshold, verbose=False)

        detections: List[Detection] = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names.get(class_id, f"class_{class_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                track_id = None
                if getattr(box, "id", None) is not None and box.id is not None:
                    track_id = int(box.id[0])

                detections.append(
                    Detection(
                        class_name=class_name.lower(),
                        class_id=class_id,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        track_id=track_id,
                    )
                )

        return detections

    def filter_target(
        self,
        detections: List[Detection],
        conf_threshold: float = 0.70,
    ) -> Tuple[List[Detection], List[Detection], List[Detection]]:
        """
        Separates detections into:
          1. Confirmed target detections (above threshold)
          2. Low-confidence target candidates (below threshold)
          3. Other object detections

        Returns:
            (confirmed, low_conf, others)
        """
        confirmed = []
        low_conf = []
        others = []

        for det in detections:
            if det.class_name == self._target_class:
                if det.confidence >= conf_threshold:
                    confirmed.append(det)
                else:
                    low_conf.append(det)
            else:
                others.append(det)

        return confirmed, low_conf, others
