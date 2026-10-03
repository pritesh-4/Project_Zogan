"""
=============================================================================
🐘 PROJECT ZOGAN — CONFIGURATION PACKAGE
=============================================================================

Central configuration package. Re-exports all settings and constants from
config.settings for convenient access across the application.

Usage:
    import config
    print(config.CONFIDENCE_THRESHOLD)

    from config import settings
    from config.settings import VILLAGE_RADIUS_METERS
=============================================================================
"""

from config.settings import *  # noqa: F401, F403
