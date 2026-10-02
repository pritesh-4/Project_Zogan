"""
=============================================================================
🐘 PROJECT ZOGAN — MODEL MANAGER
=============================================================================

Resolves, validates, and manages model paths with clean fallback logic.
Replaces scattered path resolution code that was duplicated across multiple
files.
=============================================================================
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import config

logger = logging.getLogger(__name__)


def resolve_model_path(
    use_custom: bool = True,
    override_path: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Resolves the active model path and human-readable label.

    Priority:
      1. Explicit override path (if provided)
      2. Custom model path (if use_custom=True)
      3. Fallback custom path
      4. Pretrained model path

    Returns:
        (model_path, model_label) — path may be None if not found
    """
    if override_path:
        p = Path(override_path)
        if p.exists():
            return str(p), f"Custom ({p.name})"
        logger.warning("Override model path not found: %s", override_path)
        return str(p), f"Custom ({p.name})"

    if use_custom:
        primary = Path(config.CUSTOM_MODEL_PATH)
        fallback = Path(config.FALLBACK_CUSTOM_PATH)

        if primary.exists():
            return str(primary), "elephant_v1 (Custom)"
        if fallback.exists():
            return str(fallback), "elephant_v1 (Custom)"

        logger.warning(
            "Custom model not found at '%s' or '%s'",
            config.CUSTOM_MODEL_PATH,
            config.FALLBACK_CUSTOM_PATH,
        )
        return None, "elephant_v1 (Custom)"

    pretrained = Path(config.PRETRAINED_MODEL_PATH)
    if pretrained.exists():
        return str(pretrained), "yolo26n (Pretrained)"

    logger.warning("Pretrained model not found: %s", config.PRETRAINED_MODEL_PATH)
    return str(pretrained), "yolo26n (Pretrained)"


def verify_model(model_path: str) -> bool:
    """
    Verifies that a model file exists and is a non-empty file.

    Returns:
        True if the model file is valid
    """
    p = Path(model_path)
    if not p.exists():
        logger.error("Model file does not exist: %s", model_path)
        return False
    if not p.is_file():
        logger.error("Model path is not a file: %s", model_path)
        return False
    if p.stat().st_size == 0:
        logger.error("Model file is empty: %s", model_path)
        return False
    return True
