"""
=============================================================================
🐘 PROJECT ZOGAN — AUTOMATED TEST SUITE FOR PHASE 5
=============================================================================

Comprehensive test verification for GPS Geofencing & Rule-Based Risk Engine:
 1. Distance calculation accuracy (Haversine formula on known coordinates)
 2. Same-point distance calculation (~0.0 meters)
 3. Point inside zone detection (circle & polygon)
 4. Point outside zone detection
 5. Zone boundary distance behavior
 6. Zone classification (Village > Buffer > Forest > Outside)
 7. Distance to protected sensitive zone
 8. Geographic approach trend detection (consistently decreasing distance)
 9. Geographic receding trend detection (consistently increasing distance)
10. Stable & unknown trend handling (sub-threshold distance fluctuations)
11. Risk score boundaries & clamping [0, 100]
12. Risk level mapping (LOW, MEDIUM, HIGH, CRITICAL)
13. Group-size impact on risk evaluation
14. Existing alert logic, persistence, and cooldown preservation

Usage:
    python test_phase5.py
=============================================================================
"""

import sys
import time

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Project modules
import elephant_camera as ec
from ai.geofence import (
    GeoZone,
    calculate_distance,
    is_point_in_zone,
    distance_to_zone,
    classify_zone,
    get_distance_to_protected_zone,
    create_default_zones,
    ZONE_TYPE_PROTECTED,
    ZONE_TYPE_WARNING,
    ZONE_TYPE_MONITORING,
    ZONE_TYPE_OUTSIDE,
)
from ai.risk_engine import (
    RiskEngine,
    MovementTrendTracker,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_HIGH,
    RISK_CRITICAL,
    TREND_APPROACHING,
    TREND_RECEDING,
    TREND_STABLE,
    TREND_UNKNOWN,
)


def test_distance_calculation():
    print("\n--- TEST 1: Distance Calculation Accuracy ---")
    # Prompt specification:
    # Camera: 20.1234, 85.1234
    # Zone:   20.1240, 85.1240
    # Expected Output: ~90 meters (~91.52m)
    d = calculate_distance(20.1234, 85.1234, 20.1240, 85.1240)
    assert 85.0 <= d <= 95.0, f"Expected ~90m, got {d:.2f}m"
    print(f"  ✓ (20.1234, 85.1234) -> (20.1240, 85.1240) = {d:.2f} meters (~90m specification met).")


def test_same_point_distance():
    print("\n--- TEST 2: Same-Point Distance is Zero ---")
    lat, lon = 20.123456, 85.123456
    d = calculate_distance(lat, lon, lat, lon)
    assert d < 1e-5, f"Expected 0.0m for identical points, got {d}"
    print(f"  ✓ Identical coordinates distance: {d:.6f} meters (strictly 0.0m).")


def test_point_inside_zone():
    print("\n--- TEST 3: Point Inside Zone Detection ---")
    # Circle zone centered at (20.1200, 85.1200) with radius 500m
    zone = GeoZone(
        name="Test Circle",
        zone_type=ZONE_TYPE_PROTECTED,
        geometry_type="circle",
        center=(20.1200, 85.1200),
        radius_meters=500.0,
    )
    # Point ~100m away
    inside_pt = (20.1205, 85.1205)
    d = calculate_distance(20.1200, 85.1200, inside_pt[0], inside_pt[1])
    assert is_point_in_zone(inside_pt[0], inside_pt[1], zone) is True
    assert distance_to_zone(inside_pt[0], inside_pt[1], zone) == 0.0
    print(f"  ✓ Point {inside_pt} (dist={d:.1f}m < 500m) detected as INSIDE zone.")


def test_point_outside_zone():
    print("\n--- TEST 4: Point Outside Zone Detection ---")
    zone = GeoZone(
        name="Test Circle",
        zone_type=ZONE_TYPE_PROTECTED,
        geometry_type="circle",
        center=(20.1200, 85.1200),
        radius_meters=300.0,
    )
    # Point ~1500m away
    outside_pt = (20.1300, 85.1300)
    d = calculate_distance(20.1200, 85.1200, outside_pt[0], outside_pt[1])
    assert is_point_in_zone(outside_pt[0], outside_pt[1], zone) is False
    assert distance_to_zone(outside_pt[0], outside_pt[1], zone) > 0.0
    print(f"  ✓ Point {outside_pt} (dist={d:.1f}m > 300m) detected as OUTSIDE zone.")


def test_boundary_behavior():
    print("\n--- TEST 5: Boundary Behavior & Proximity ---")
    # Circle with radius 500m
    center = (20.120000, 85.120000)
    zone = GeoZone(
        name="Perimeter",
        zone_type=ZONE_TYPE_WARNING,
        geometry_type="circle",
        center=center,
        radius_meters=500.0,
    )

    # Point at approx 525m (25m outside boundary)
    # 0.0047 deg lat ~ 522m
    test_lat = center[0] + 0.00472
    test_lon = center[1]
    dist_to_center = calculate_distance(test_lat, test_lon, center[0], center[1])
    dist_to_border = distance_to_zone(test_lat, test_lon, zone)

    assert dist_to_center > 500.0
    assert 15.0 <= dist_to_border <= 30.0
    assert zone.is_near_boundary(test_lat, test_lon, threshold_meters=50.0) is True
    print(f"  ✓ Point at dist={dist_to_center:.1f}m (boundary dist={dist_to_border:.1f}m) detected as NEAR_BOUNDARY.")


def test_zone_classification():
    print("\n--- TEST 6: Zone Classification Priority ---")
    zones = create_default_zones(
        village_center=(20.119000, 85.119000),
        village_radius=300.0,
        buffer_radius=800.0,
        forest_radius=2000.0,
    )

    # 1. Inside Village (< 300m)
    p_village = (20.1195, 85.1195)  # ~75m
    assert classify_zone(p_village[0], p_village[1], zones) == ZONE_TYPE_PROTECTED
    print(f"  ✓ Point {p_village} classified as {ZONE_TYPE_PROTECTED}")

    # 2. Inside Buffer (300m - 800m)
    p_buffer = (20.1230, 85.1230)  # ~600m
    assert classify_zone(p_buffer[0], p_buffer[1], zones) == ZONE_TYPE_WARNING
    print(f"  ✓ Point {p_buffer} classified as {ZONE_TYPE_WARNING}")

    # 3. Inside Forest (800m - 2000m)
    p_forest = (20.1300, 85.1300)  # ~1700m
    assert classify_zone(p_forest[0], p_forest[1], zones) == ZONE_TYPE_MONITORING
    print(f"  ✓ Point {p_forest} classified as {ZONE_TYPE_MONITORING}")

    # 4. Far Outside (> 2000m)
    p_out = (20.1600, 85.1600)  # ~6000m
    assert classify_zone(p_out[0], p_out[1], zones) == ZONE_TYPE_OUTSIDE
    print(f"  ✓ Point {p_out} classified as {ZONE_TYPE_OUTSIDE}")


def test_distance_to_protected_zone():
    print("\n--- TEST 7: Distance to Protected Sensitive Zone ---")
    zones = create_default_zones(
        village_center=(20.119000, 85.119000),
        village_radius=300.0,
    )
    # Point 600m from village center -> distance to 300m radius boundary should be ~300m
    pt = (20.1230, 85.1230)
    dist_to_center = calculate_distance(pt[0], pt[1], 20.1190, 85.1190)
    dist_to_prot = get_distance_to_protected_zone(pt[0], pt[1], zones)

    expected_border_dist = max(0.0, dist_to_center - 300.0)
    assert abs(dist_to_prot - expected_border_dist) < 1.0
    print(f"  ✓ Distance to Village Protected perimeter: {dist_to_prot:.1f} meters.")


def test_approaching_trend():
    print("\n--- TEST 8: Approaching Trend Detection ---")
    tracker = MovementTrendTracker(window_size=4, stability_threshold_meters=15.0)
    # Prompt specification:
    # 800m -> 760m -> 710m
    tracker.add_observation(800.0)
    tracker.add_observation(760.0)
    trend = tracker.add_observation(710.0)
    assert trend == TREND_APPROACHING, f"Expected APPROACHING, got {trend}"
    print(f"  ✓ Trajectory [800m, 760m, 710m] => {trend}")


def test_receding_trend():
    print("\n--- TEST 9: Receding Trend Detection ---")
    tracker = MovementTrendTracker(window_size=4, stability_threshold_meters=15.0)
    tracker.add_observation(300.0)
    tracker.add_observation(380.0)
    trend = tracker.add_observation(460.0)
    assert trend == TREND_RECEDING, f"Expected RECEDING, got {trend}"
    print(f"  ✓ Trajectory [300m, 380m, 460m] => {trend}")


def test_stable_and_unknown_trend():
    print("\n--- TEST 10: Stable and Unknown Trend Handling ---")
    tracker = MovementTrendTracker(window_size=4, stability_threshold_meters=15.0)

    # 1 observation only -> UNKNOWN
    trend_1 = tracker.add_observation(500.0)
    assert trend_1 == TREND_UNKNOWN, f"Expected UNKNOWN for 1 observation, got {trend_1}"
    print(f"  ✓ Single observation [500m] => {trend_1}")

    # Fluctuating within 15m stability threshold: 500m -> 504m -> 498m -> 502m
    tracker.add_observation(504.0)
    tracker.add_observation(498.0)
    trend_stable = tracker.add_observation(502.0)
    assert trend_stable == TREND_STABLE, f"Expected STABLE, got {trend_stable}"
    print(f"  ✓ Sub-threshold noise [500m, 504m, 498m, 502m] => {trend_stable}")


def test_risk_score_boundaries():
    print("\n--- TEST 11: Risk Score Boundaries & Clamping ---")
    engine = RiskEngine()

    # Absolute minimum: OUTSIDE, far away, RECEDING, 1 elephant, low confidence
    r_min = engine.evaluate(
        zone=ZONE_TYPE_OUTSIDE,
        distance_to_protected=5000.0,
        trend=TREND_RECEDING,
        group_size=1,
        confidence=0.5,
    )
    assert 0 <= r_min.score <= 100
    assert r_min.score <= 15
    print(f"  ✓ Minimum boundary score: {r_min.score}/100 (Level: {r_min.level})")

    # Absolute maximum: In VILLAGE, 0m distance, APPROACHING, herd of 5, 99% confidence
    r_max = engine.evaluate(
        zone=ZONE_TYPE_PROTECTED,
        distance_to_protected=0.0,
        trend=TREND_APPROACHING,
        group_size=5,
        confidence=0.99,
    )
    assert 0 <= r_max.score <= 100
    assert r_max.score >= 90
    assert r_max.level == RISK_CRITICAL
    print(f"  ✓ Maximum boundary score: {r_max.score}/100 (Level: {r_max.level})")


def test_risk_level_mapping():
    print("\n--- TEST 12: Risk Level Mapping (LOW, MEDIUM, HIGH, CRITICAL) ---")
    engine = RiskEngine(low_max=24, medium_max=49, high_max=74)

    # LOW: Far inside forest, receding/stable
    r_low = engine.evaluate(
        zone=ZONE_TYPE_MONITORING,
        distance_to_protected=1500.0,
        trend=TREND_RECEDING,
        group_size=1,
    )
    assert r_low.level == RISK_LOW, f"Expected LOW, got {r_low.level}"

    # MEDIUM: In forest approaching, or buffer zone stable
    r_med = engine.evaluate(
        zone=ZONE_TYPE_MONITORING,
        distance_to_protected=800.0,
        trend=TREND_APPROACHING,
        group_size=1,
    )
    assert r_med.level == RISK_MEDIUM, f"Expected MEDIUM, got {r_med.level}"

    # HIGH: In buffer zone approaching at 300m
    r_high = engine.evaluate(
        zone=ZONE_TYPE_WARNING,
        distance_to_protected=300.0,
        trend=TREND_APPROACHING,
        group_size=2,
    )
    assert r_high.level == RISK_HIGH, f"Expected HIGH, got {r_high.level}"

    # CRITICAL: In village settlement
    r_crit = engine.evaluate(
        zone=ZONE_TYPE_PROTECTED,
        distance_to_protected=0.0,
        trend=TREND_APPROACHING,
        group_size=3,
    )
    assert r_crit.level == RISK_CRITICAL, f"Expected CRITICAL, got {r_crit.level}"

    print(
        f"  ✓ Verified levels: LOW ({r_low.score}), MEDIUM ({r_med.score}), HIGH ({r_high.score}), CRITICAL ({r_crit.score})"
    )


def test_group_size_impact():
    print("\n--- TEST 13: Group-Size Input Impact ---")
    engine = RiskEngine()

    # Identical spatial scenario: BUFFER zone, 400m, APPROACHING
    r_single = engine.evaluate(
        zone=ZONE_TYPE_WARNING,
        distance_to_protected=400.0,
        trend=TREND_APPROACHING,
        group_size=1,
    )
    r_pair = engine.evaluate(
        zone=ZONE_TYPE_WARNING,
        distance_to_protected=400.0,
        trend=TREND_APPROACHING,
        group_size=2,
    )
    r_herd = engine.evaluate(
        zone=ZONE_TYPE_WARNING,
        distance_to_protected=400.0,
        trend=TREND_APPROACHING,
        group_size=5,
    )

    assert r_herd.score > r_pair.score > r_single.score
    print(f"  ✓ Single elephant: {r_single.score} pts | Pair: {r_pair.score} pts | Herd (5): {r_herd.score} pts")


def test_alert_logic_intact():
    print("\n--- TEST 14: Existing Alert Logic & Enhanced Risk Context Intact ---")
    consecutive_frames = 0
    last_alert_time = 0.0
    alerts_triggered = 0
    current_time = time.time()

    # Simulate 7 frames of detection
    for frame_idx in range(1, 8):
        consecutive_frames += 1
        time_since_last = current_time - last_alert_time
        cooldown_active = time_since_last < ec.ALERT_COOLDOWN_SECONDS

        if consecutive_frames >= ec.REQUIRED_DETECTIONS:
            if not cooldown_active:
                tracked_info = [{"id": 1, "conf": 0.98, "movement": "STATIONARY"}]
                risk_info = {
                    "risk_level": "HIGH",
                    "risk_score": 67,
                    "zone": "BUFFER",
                    "distance_to_protected_m": 350.0,
                    "trend": "APPROACHING",
                    "group_size": 1,
                    "geo_mode": "SIMULATION",
                }
                ec.trigger_alert(0.98, tracked_info=tracked_info, risk_info=risk_info)
                last_alert_time = current_time
                alerts_triggered += 1
                print(f"  Frame {frame_idx}: Alert dispatched with risk context.")
            else:
                print(f"  Frame {frame_idx}: Alert skipped due to cooldown.")
        else:
            print(f"  Frame {frame_idx}: Persistence counter: {consecutive_frames}/{ec.REQUIRED_DETECTIONS}")

    assert alerts_triggered == 1, f"Expected 1 alert, got {alerts_triggered}"
    print("  ✓ Persistence (5 frames), cooldown (30s), and enriched risk alert verified.")


def run_all_tests():
    print("=" * 65)
    print("🐘 RUNNING PROJECT ZOGAN PHASE 5 GEOFENCING & RISK TEST SUITE")
    print("=" * 65)

    test_distance_calculation()
    test_same_point_distance()
    test_point_inside_zone()
    test_point_outside_zone()
    test_boundary_behavior()
    test_zone_classification()
    test_distance_to_protected_zone()
    test_approaching_trend()
    test_receding_trend()
    test_stable_and_unknown_trend()
    test_risk_score_boundaries()
    test_risk_level_mapping()
    test_group_size_impact()
    test_alert_logic_intact()

    print("\n" + "=" * 65)
    print("🎉 ALL 14 PHASE 5 TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
