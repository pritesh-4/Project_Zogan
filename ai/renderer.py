"""
=============================================================================
🐘 PROJECT ZOGAN — HUD & VISUAL RENDERING
=============================================================================

UI rendering functions for bounding boxes, heads-up display, and alert
banners. These consume system state but do NOT define business logic.
=============================================================================
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

import config


def draw_bounding_box(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: Tuple[int, int, int],
    is_target: bool = False,
    sub_label: Optional[str] = None,
    trajectory: Optional[List[Tuple[float, float]]] = None,
    hud_bar_height: int = 85,
) -> None:
    """
    Draws a styled bounding box with an informative background label tag.
    Optionally renders movement direction sub-label, center point, and trajectory trail.
    """
    thickness = 3 if is_target else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw center point and trajectory trail if available
    if trajectory and len(trajectory) > 1:
        pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [pts],
            isClosed=False,
            color=(0, 215, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if trajectory and len(trajectory) > 0:
        cx, cy = int(trajectory[-1][0]), int(trajectory[-1][1])
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1, cv2.LINE_AA)

    # Calculate text sizes for multi-line badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55 if is_target else 0.5
    text_thickness = 2 if is_target else 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)

    sub_w, sub_h = 0, 0
    if sub_label:
        sub_scale = 0.45
        (sub_w, sub_h), _ = cv2.getTextSize(sub_label, font, sub_scale, 1)

    badge_w = max(text_w, sub_w) + 14
    line_spacing = 4
    badge_height = text_h + (sub_h + line_spacing if sub_label else 0) + 12

    # Avoid top HUD bar
    if y1 - badge_height >= hud_bar_height:
        label_y1 = y1 - badge_height
        label_y2 = y1
    else:
        label_y1 = max(y1, hud_bar_height + 4)
        label_y2 = label_y1 + badge_height

    label_x1 = max(0, x1)
    label_x2 = min(frame.shape[1], label_x1 + badge_w)

    # Draw label background rectangle
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), 1)

    # Primary label
    t1_y = label_y1 + text_h + 5
    cv2.putText(
        frame,
        label,
        (label_x1 + 6, t1_y),
        font,
        font_scale,
        config.COLOR_TEXT_WHITE,
        text_thickness,
        cv2.LINE_AA,
    )

    # Movement sub-label
    if sub_label:
        t2_y = t1_y + sub_h + line_spacing
        cv2.putText(
            frame,
            sub_label,
            (label_x1 + 6, t2_y),
            font,
            0.45,
            (220, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_hud(
    frame,
    status_text: str,
    status_color: Tuple[int, int, int],
    detection_count: int,
    fps: float,
    cooldown_remaining: float,
    model_name: str = "",
    tracking_summary: str = "",
    tracked_count: int = 0,
    risk_assessment=None,
    geo_mode: str = "SIMULATION",
) -> None:
    """
    Draws an informative Heads-Up Display (HUD) overlay at the top of the video frame.
    """
    from ai.risk_engine import RISK_CRITICAL, RISK_HIGH, RISK_MEDIUM

    height, width = frame.shape[:2]

    has_tracks = (tracked_count > 0) and bool(tracking_summary)
    has_risk = risk_assessment is not None and tracked_count > 0

    if has_risk:
        bar_height = 98
    elif has_tracks:
        bar_height = 80
    else:
        bar_height = 65

    # Create top status bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_height), config.COLOR_OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Determine responsive font scale based on frame width
    scale_factor = min(1.0, max(0.7, width / 700.0))
    status_scale = 0.68 * scale_factor
    sub_scale = 0.48 * scale_factor

    # Row 1: Status text (Left) & Model tag (Right)
    status_str = f"STATUS: {status_text}"
    cv2.putText(
        frame,
        status_str,
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        status_scale,
        status_color,
        2,
        cv2.LINE_AA,
    )

    model_tag = f"MODEL: {model_name}"
    (model_w, _), _ = cv2.getTextSize(model_tag, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
    model_x = max(int(width * 0.52), width - model_w - 15)
    cv2.putText(
        frame,
        model_tag,
        (model_x, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_scale,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Row 2: Persistence / Cooldown info & FPS
    if 0 < detection_count < config.REQUIRED_DETECTIONS:
        sub_text = f"Persistence: {detection_count}/{config.REQUIRED_DETECTIONS} consecutive frames"
        sub_color = config.COLOR_WARN_YELLOW
    elif cooldown_remaining > 0:
        sub_text = f"Cooldown Active: {cooldown_remaining:.0f}s remaining (Monitoring continues)"
        sub_color = config.COLOR_WARN_YELLOW
    else:
        sub_text = f"Target: {config.TARGET_CLASS.upper()} | Min Conf: {int(config.CONFIDENCE_THRESHOLD * 100)}% | Press 'Q' to exit"
        sub_color = (200, 200, 200)

    cv2.putText(
        frame,
        sub_text,
        (15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_scale,
        sub_color,
        1,
        cv2.LINE_AA,
    )

    fps_text = f"FPS: {fps:.1f}"
    (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
    cv2.putText(
        frame,
        fps_text,
        (width - fps_w - 15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Row 3: Global Tracking HUD summary
    if has_tracks:
        track_text = f"TRACKS ({tracked_count}): {tracking_summary}"
        cv2.putText(
            frame,
            track_text,
            (15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            sub_scale,
            config.COLOR_WARN_YELLOW,
            1,
            cv2.LINE_AA,
        )

    # Row 4: Geofencing & Risk Assessment Summary
    if has_risk:
        r = risk_assessment
        risk_color = (
            config.COLOR_ALERT_RED
            if r.level in (RISK_CRITICAL, RISK_HIGH)
            else (config.COLOR_WARN_YELLOW if r.level == RISK_MEDIUM else config.COLOR_SAFE_GREEN)
        )
        geo_text = (
            f"GEO [{geo_mode}]: Zone: {r.zone} | Protected Dist: {r.distance_to_protected:.0f}m | "
            f"Trend: {r.trend} | Risk: {r.level} ({r.score}/100)"
        )
        cv2.putText(
            frame,
            geo_text,
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43 * scale_factor,
            risk_color,
            1,
            cv2.LINE_AA,
        )


def draw_alert_banner(frame, risk_level: Optional[str] = None) -> None:
    """Renders an emergency visual alert banner across the bottom of the screen."""
    height, width = frame.shape[:2]
    banner_y1 = height - 70
    banner_y2 = height - 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (20, banner_y1), (width - 20, banner_y2), config.COLOR_ALERT_RED, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.rectangle(frame, (20, banner_y1), (width - 20, banner_y2), config.COLOR_TEXT_WHITE, 2)

    tag = f" — RISK: {risk_level}" if risk_level else ""
    alert_msg = f"[!] ELEPHANT DETECTED — EARLY WARNING ALERT{tag} [!]"
    (text_w, text_h), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    text_x = max(30, (width - text_w) // 2)
    cv2.putText(
        frame,
        alert_msg,
        (text_x, banner_y1 + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        config.COLOR_TEXT_WHITE,
        2,
        cv2.LINE_AA,
    )
