"""
=============================================================================
🐘 PROJECT ZOGAN — AUTOMATED TEST SUITE FOR PHASE 7
INTELLIGENT THREAT ASSESSMENT & RISK ENGINE 2.0
=============================================================================

Comprehensive test verification across 20 core evaluation scenarios:
  1. Low-risk detection
  2. Medium-risk detection
  3. High-risk detection
  4. Critical-risk detection
  5. Single-frame false positive
  6. Persistent detection
  7. Elephant leaving zone
  8. Elephant entering zone
  9. Elephant approaching relevant boundary
 10. Stationary elephant
 11. Multiple elephants
 12. Risk decreasing after evidence disappears
 13. Event resolution
 14. Alert cooldown
 15. Alert escalation
 16. Deterministic scoring
 17. Invalid/missing inputs
 18. Extreme confidence values
 19. Score clamping
 20. Configuration overrides

Safety & Architecture Rules:
  - No GPU, live camera, physical GPS, or Telegram network calls required.
  - Fully mockable and deterministic.
=============================================================================
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.geofence import (
    ZONE_TYPE_MONITORING,
    ZONE_TYPE_OUTSIDE,
    ZONE_TYPE_PROTECTED,
    ZONE_TYPE_WARNING,
)
from ai.risk_engine import (
    EVENT_STATE_CONFIRMED,
    EVENT_STATE_DETECTED,
    EVENT_STATE_HIGH_RISK,
    EVENT_STATE_RESOLVED,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    TREND_APPROACHING,
    TREND_RECEDING,
    TREND_STABLE,
    AlertPolicy,
    EventLifecycleManager,
    RiskEngine,
)


def test_1_low_risk_detection():
    print("\n--- TEST 1: Low-Risk Detection ---")
    engine = RiskEngine()
    policy = AlertPolicy()

    # Distant forest location, receding from protected boundary, solitary
    assessment = engine.evaluate(
        zone=ZONE_TYPE_MONITORING,
        distance_to_protected=1600.0,
        trend=TREND_RECEDING,
        group_size=1,
        confidence=0.75,
        persistence_frames=6,
    )

    assert assessment.level == RISK_LOW, f"Expected LOW risk, got {assessment.level}"
    assert 0 <= assessment.score <= 24, f"Expected score 0-24, got {assessment.score}"
    assert assessment.alert_recommended is False
    assert len(assessment.reasons) >= 3

    # Alert policy evaluation
    decision = policy.evaluate(
        assessment=assessment,
        current_time=100.0,
        last_alert_time=0.0,
    )
    assert decision.should_dispatch is False
    assert "log" in decision.delivery_channels
    print(f"  ✓ LOW risk verified: Score {assessment.score}/100 | Reasons: {len(assessment.reasons)}")


def test_2_medium_risk_detection():
    print("\n--- TEST 2: Medium-Risk Detection ---")
    engine = RiskEngine()
    policy = AlertPolicy()

    # Forest perimeter, stable distance (750m), moderate confidence
    assessment = engine.evaluate(
        zone=ZONE_TYPE_MONITORING,
        distance_to_protected=750.0,
        trend=TREND_STABLE,
        group_size=1,
        confidence=0.85,
        persistence_frames=6,
    )

    assert assessment.level == RISK_MEDIUM, f"Expected MEDIUM risk, got {assessment.level}"
    assert 25 <= assessment.score <= 49, f"Expected score 25-49, got {assessment.score}"
    assert assessment.alert_recommended is False

    decision = policy.evaluate(
        assessment=assessment,
        current_time=100.0,
        last_alert_time=0.0,
    )
    assert decision.should_dispatch is False
    assert "hud" in decision.delivery_channels
    print(f"  ✓ MEDIUM risk verified: Score {assessment.score}/100 | Level: {assessment.level}")


def test_3_high_risk_detection():
    print("\n--- TEST 3: High-Risk Detection ---")
    engine = RiskEngine()
    policy = AlertPolicy()

    # Buffer zone, 350m to village boundary, approaching, 2 elephants
    assessment = engine.evaluate(
        zone=ZONE_TYPE_WARNING,
        distance_to_protected=350.0,
        trend=TREND_APPROACHING,
        group_size=2,
        confidence=0.92,
        persistence_frames=7,
    )

    assert assessment.level == RISK_HIGH, f"Expected HIGH risk, got {assessment.level}"
    assert 50 <= assessment.score <= 74, f"Expected score 50-74, got {assessment.score}"
    assert assessment.alert_recommended is True

    decision = policy.evaluate(
        assessment=assessment,
        current_time=100.0,
        last_alert_time=0.0,
    )
    assert decision.should_dispatch is True
    assert "telegram" in decision.delivery_channels
    print(f"  ✓ HIGH risk verified: Score {assessment.score}/100 | Dispatch: {decision.should_dispatch}")


def test_4_critical_risk_detection():
    print("\n--- TEST 4: Critical-Risk Detection ---")
    engine = RiskEngine()
    policy = AlertPolicy()

    # Inside protected village settlement, 0m distance, approaching, high confidence
    assessment = engine.evaluate(
        zone=ZONE_TYPE_PROTECTED,
        distance_to_protected=0.0,
        trend=TREND_APPROACHING,
        group_size=3,
        confidence=0.95,
        persistence_frames=8,
    )

    assert assessment.level == RISK_CRITICAL, f"Expected CRITICAL risk, got {assessment.level}"
    assert assessment.score >= 75, f"Expected score >= 75, got {assessment.score}"
    assert assessment.alert_recommended is True

    decision = policy.evaluate(
        assessment=assessment,
        current_time=100.0,
        last_alert_time=0.0,
    )
    assert decision.should_dispatch is True
    assert "telegram" in decision.delivery_channels
    print(f"  ✓ CRITICAL risk verified: Score {assessment.score}/100 | Level: {assessment.level}")


def test_5_single_frame_false_positive():
    print("\n--- TEST 5: Single-Frame False Positive Suppression ---")
    engine = RiskEngine()
    policy = AlertPolicy()

    # Noisy detection inside protected zone with high confidence, BUT persistence is only 1 frame
    assessment = engine.evaluate(
        zone=ZONE_TYPE_PROTECTED,
        distance_to_protected=0.0,
        trend=TREND_APPROACHING,
        group_size=1,
        confidence=0.99,
        persistence_frames=1,
    )

    # Critical safety rule: unconfirmed detection must NOT produce CRITICAL or trigger alert
    assert assessment.level != RISK_CRITICAL, f"Single frame must not be CRITICAL, got {assessment.level}"
    assert assessment.level == RISK_MEDIUM, f"Expected capped at MEDIUM, got {assessment.level}"
    assert assessment.alert_recommended is False, "Unconfirmed single frame should not recommend alert"

    # Verify explanatory reason
    has_safeguard_reason = any("Evidence safeguard" in r for r in assessment.reasons)
    assert has_safeguard_reason, "Expected safeguard reason for unconfirmed detection"

    # Alert policy suppresses dispatch
    decision = policy.evaluate(assessment, current_time=100.0, last_alert_time=0.0)
    assert decision.should_dispatch is False
    print(f"  ✓ False positive suppressed: Level capped at {assessment.level} | Alert withheld")


def test_6_persistent_detection():
    print("\n--- TEST 6: Persistent Detection Evidence Accumulation ---")
    engine = RiskEngine()

    # Location evaluated with increasing persistence across evaluation window
    a_f1 = engine.evaluate(ZONE_TYPE_MONITORING, 800.0, TREND_APPROACHING, 1, 0.90, persistence_frames=1)
    a_f3 = engine.evaluate(ZONE_TYPE_MONITORING, 800.0, TREND_APPROACHING, 1, 0.90, persistence_frames=3)
    a_f6 = engine.evaluate(ZONE_TYPE_MONITORING, 800.0, TREND_APPROACHING, 1, 0.90, persistence_frames=6)
    a_f12 = engine.evaluate(ZONE_TYPE_MONITORING, 800.0, TREND_APPROACHING, 1, 0.90, persistence_frames=12)

    assert a_f1.contributing_factors["persistence_points"] == 0
    assert a_f3.contributing_factors["persistence_points"] == 5
    assert a_f6.contributing_factors["persistence_points"] == 10
    assert a_f12.contributing_factors["persistence_points"] == 15
    assert a_f12.score > a_f6.score > a_f3.score > a_f1.score
    print(
        f"  ✓ Persistence evidence: 1 frame={a_f1.score} | 3 frames={a_f3.score} | "
        f"6 frames={a_f6.score} | 12 frames={a_f12.score}"
    )


def test_7_elephant_leaving_zone():
    print("\n--- TEST 7: Elephant Leaving Zone (Receding Outward) ---")
    engine = RiskEngine()

    # Stepwise retreat: VILLAGE -> BUFFER -> FOREST -> OUTSIDE
    step1 = engine.evaluate(ZONE_TYPE_PROTECTED, 0.0, TREND_RECEDING, 1, 0.90, persistence_frames=6)
    step2 = engine.evaluate(ZONE_TYPE_WARNING, 350.0, TREND_RECEDING, 1, 0.90, persistence_frames=7)
    step3 = engine.evaluate(ZONE_TYPE_MONITORING, 900.0, TREND_RECEDING, 1, 0.90, persistence_frames=8)
    step4 = engine.evaluate(ZONE_TYPE_OUTSIDE, 2500.0, TREND_RECEDING, 1, 0.90, persistence_frames=9)

    assert step1.score > step2.score > step3.score > step4.score
    assert step4.level == RISK_LOW
    assert step2.contributing_factors["trend_points"] == 0  # RECEDING yields 0 trend points
    print(
        f"  ✓ Outward trajectory scores: Village={step1.score} -> Buffer={step2.score} -> "
        f"Forest={step3.score} -> Outside={step4.score} (Level: {step4.level})"
    )


def test_8_elephant_entering_zone():
    print("\n--- TEST 8: Elephant Entering Zone (Inward Incursion) ---")
    engine = RiskEngine()

    # Stepwise incursion: OUTSIDE -> FOREST -> BUFFER -> VILLAGE
    step1 = engine.evaluate(ZONE_TYPE_OUTSIDE, 2500.0, TREND_APPROACHING, 1, 0.90, persistence_frames=6)
    step2 = engine.evaluate(ZONE_TYPE_MONITORING, 900.0, TREND_APPROACHING, 1, 0.90, persistence_frames=7)
    step3 = engine.evaluate(ZONE_TYPE_WARNING, 350.0, TREND_APPROACHING, 1, 0.90, persistence_frames=8)
    step4 = engine.evaluate(ZONE_TYPE_PROTECTED, 0.0, TREND_APPROACHING, 1, 0.90, persistence_frames=9)

    assert step1.score < step2.score < step3.score < step4.score
    assert step4.level == RISK_CRITICAL
    print(
        f"  ✓ Inward trajectory scores: Outside={step1.score} -> Forest={step2.score} -> "
        f"Buffer={step3.score} -> Village={step4.score} (Level: {step4.level})"
    )


def test_9_elephant_approaching_boundary():
    print("\n--- TEST 9: Elephant Approaching Protected Boundary ---")
    engine = RiskEngine()

    # Fix buffer zone, observe closing distance
    d700 = engine.evaluate(ZONE_TYPE_WARNING, 700.0, TREND_APPROACHING, 1, 0.90, persistence_frames=6)
    d350 = engine.evaluate(ZONE_TYPE_WARNING, 350.0, TREND_APPROACHING, 1, 0.90, persistence_frames=7)
    d100 = engine.evaluate(ZONE_TYPE_WARNING, 100.0, TREND_APPROACHING, 1, 0.90, persistence_frames=8)

    assert d100.contributing_factors["proximity_points"] > d350.contributing_factors["proximity_points"]
    assert d350.contributing_factors["proximity_points"] > d700.contributing_factors["proximity_points"]
    assert d100.score > d350.score > d700.score
    print(f"  ✓ Boundary approach: 700m={d700.score} pts -> 350m={d350.score} pts -> 100m={d100.score} pts")


def test_10_stationary_elephant():
    print("\n--- TEST 10: Stationary Elephant Evaluation ---")
    engine = RiskEngine()

    assessment = engine.evaluate(
        zone=ZONE_TYPE_WARNING,
        distance_to_protected=400.0,
        trend=TREND_STABLE,
        group_size=1,
        confidence=0.88,
        persistence_frames=6,
    )

    assert assessment.contributing_factors["trend_points"] == 5
    assert any("STABLE" in r for r in assessment.reasons)
    print(f"  ✓ Stationary target verified: Score {assessment.score} | Trend: STABLE (5 pts)")


def test_11_multiple_elephants():
    print("\n--- TEST 11: Multiple Elephants (Herd Context) ---")
    engine = RiskEngine()

    tracks_solo = [1]
    tracks_pair = [1, 2]
    tracks_herd = [1, 2, 3, 4, 5]

    a_solo = engine.evaluate(ZONE_TYPE_WARNING, 400.0, TREND_APPROACHING, 1, 0.90, tracked_objects=tracks_solo)
    a_pair = engine.evaluate(ZONE_TYPE_WARNING, 400.0, TREND_APPROACHING, 2, 0.90, tracked_objects=tracks_pair)
    a_herd = engine.evaluate(ZONE_TYPE_WARNING, 400.0, TREND_APPROACHING, 5, 0.90, tracked_objects=tracks_herd)

    assert a_herd.score > a_pair.score > a_solo.score
    assert a_herd.tracked_objects == [1, 2, 3, 4, 5]
    assert any("Large herd" in r for r in a_herd.reasons)
    print(f"  ✓ Herd evaluation: Solo={a_solo.score} pts | Pair={a_pair.score} pts | Herd={a_herd.score} pts")


def test_12_risk_decreasing_after_evidence_disappears():
    print("\n--- TEST 12: Risk Decreasing when Evidence Subsides ---")
    engine = RiskEngine()

    # Peak threat
    peak = engine.evaluate(ZONE_TYPE_PROTECTED, 0.0, TREND_APPROACHING, 2, 0.95, persistence_frames=10)
    assert peak.level == RISK_CRITICAL

    # Target turns around and recedes
    subsiding = engine.evaluate(ZONE_TYPE_WARNING, 450.0, TREND_RECEDING, 1, 0.85, persistence_frames=15)
    assert subsiding.score < peak.score
    assert subsiding.level in (RISK_MEDIUM, RISK_HIGH)

    # Target enters forest and recedes further
    cleared = engine.evaluate(ZONE_TYPE_MONITORING, 1500.0, TREND_RECEDING, 1, 0.75, persistence_frames=2)
    assert cleared.score < subsiding.score
    assert cleared.level == RISK_LOW
    print(f"  ✓ De-escalation verified: Peak={peak.level} -> Subsiding={subsiding.level} -> Cleared={cleared.level}")


def test_13_event_resolution():
    print("\n--- TEST 13: Event Lifecycle Resolution ---")
    manager = EventLifecycleManager(confirmation_frames=5, resolution_frames=10, resolution_seconds=1.0)
    t = 1000.0

    # 1. New event emerges
    trans = manager.update(elephant_detected=True, current_time=t)
    assert trans.transition_type == "NEW_EVENT"
    assert trans.state == EVENT_STATE_DETECTED
    initial_id = trans.event_id
    assert initial_id is not None

    # 2. Confirmed presence
    for _ in range(5):
        t += 0.1
        trans = manager.update(elephant_detected=True, current_time=t)
    assert trans.state in (EVENT_STATE_CONFIRMED, EVENT_STATE_HIGH_RISK)

    # 3. Elephant missing for 10 frames -> RESOLVED
    resolved_trans = None
    for _ in range(10):
        t += 0.2
        res = manager.update(elephant_detected=False, current_time=t)
        if res.transition_type == "RESOLVED":
            resolved_trans = res
            break

    assert resolved_trans is not None, "Event should resolve after 10 missed frames"
    assert resolved_trans.resolved_event_id == initial_id
    assert resolved_trans.state == EVENT_STATE_RESOLVED

    # 4. Next detection creates a brand NEW event ID
    t += 5.0
    new_trans = manager.update(elephant_detected=True, current_time=t)
    assert new_trans.transition_type == "NEW_EVENT"
    assert new_trans.event_id != initial_id
    print(f"  ✓ Event lifecycle: Initial ID '{initial_id}' resolved cleanly; New ID '{new_trans.event_id}' spawned.")


def test_14_alert_cooldown():
    print("\n--- TEST 14: Alert Cooldown Suppression ---")
    policy = AlertPolicy(cooldown_seconds=30.0, allow_escalation_bypass=True)
    engine = RiskEngine()

    assessment = engine.evaluate(ZONE_TYPE_WARNING, 300.0, TREND_APPROACHING, 1, 0.90, persistence_frames=6)

    # t=100.0: Initial alert fires
    d1 = policy.evaluate(assessment, current_time=100.0, last_alert_time=0.0)
    assert d1.should_dispatch is True

    # t=110.0 (10s later, cooldown active): Alert suppressed
    d2 = policy.evaluate(assessment, current_time=110.0, last_alert_time=100.0, last_alert_level=assessment.level)
    assert d2.should_dispatch is False
    assert "suppressed by cooldown" in d2.reason

    # t=135.0 (35s later, cooldown expired): Alert allowed again
    d3 = policy.evaluate(assessment, current_time=135.0, last_alert_time=100.0, last_alert_level=assessment.level)
    assert d3.should_dispatch is True
    print("  ✓ Cooldown: Initial alert allowed -> 10s alert suppressed -> 35s alert allowed.")


def test_15_alert_escalation():
    print("\n--- TEST 15: Evidence-Based Alert Escalation (Cooldown Bypass) ---")
    policy = AlertPolicy(cooldown_seconds=30.0, allow_escalation_bypass=True)
    engine = RiskEngine()

    # t=100.0: Alerted at HIGH risk in buffer zone
    a_high = engine.evaluate(ZONE_TYPE_WARNING, 400.0, TREND_APPROACHING, 1, 0.90, persistence_frames=6)
    assert a_high.level == RISK_HIGH

    # t=105.0: Target suddenly enters VILLAGE settlement -> CRITICAL!
    a_crit = engine.evaluate(ZONE_TYPE_PROTECTED, 0.0, TREND_APPROACHING, 2, 0.95, persistence_frames=7)
    assert a_crit.level == RISK_CRITICAL

    # Policy must bypass cooldown because level escalated from HIGH to CRITICAL
    decision = policy.evaluate(
        assessment=a_crit,
        current_time=105.0,
        last_alert_time=100.0,
        last_alert_level=RISK_HIGH,
        is_escalated=True,
    )

    assert decision.should_dispatch is True
    assert decision.cooldown_bypassed is True
    assert decision.is_escalation is True
    assert "Cooldown bypassed" in decision.reason
    print("  ✓ Escalation: HIGH -> CRITICAL bypassed active cooldown to dispatch immediate alarm.")


def test_16_deterministic_scoring():
    print("\n--- TEST 16: Deterministic Scoring Consistency ---")
    engine = RiskEngine()

    scores = []
    levels = []
    breakdowns = []

    for _ in range(100):
        res = engine.evaluate(
            zone=ZONE_TYPE_WARNING,
            distance_to_protected=280.0,
            trend=TREND_APPROACHING,
            group_size=2,
            confidence=0.88,
            persistence_frames=6,
        )
        scores.append(res.score)
        levels.append(res.level)
        breakdowns.append(res.breakdown)

    assert all(s == scores[0] for s in scores), "Scores must be strictly deterministic"
    assert all(lvl == levels[0] for lvl in levels), "Levels must be strictly deterministic"
    assert all(b == breakdowns[0] for b in breakdowns), "Breakdowns must be strictly deterministic"
    print(f"  ✓ Determinism verified: 100 iterations produced identical score {scores[0]} and level {levels[0]}.")


def test_17_invalid_missing_inputs():
    print("\n--- TEST 17: Invalid and Missing Inputs Resilience ---")
    engine = RiskEngine()

    # Engine must not raise unhandled exceptions on invalid or malformed data
    res_neg_dist = engine.evaluate(zone=ZONE_TYPE_WARNING, distance_to_protected=-50.0)
    assert res_neg_dist.distance_to_protected == 0.0

    res_unknown_zone = engine.evaluate(zone="UNKNOWN_NONEXISTENT_ZONE", distance_to_protected=500.0)
    assert res_unknown_zone.zone == "UNKNOWN_NONEXISTENT_ZONE"
    assert res_unknown_zone.contributing_factors["zone_points"] == 0

    res_none_vals = engine.evaluate(zone=None, distance_to_protected=float("inf"), trend=None, confidence=None)
    assert res_none_vals.level in (RISK_LOW, RISK_MEDIUM)

    res_empty_tracks = engine.evaluate(ZONE_TYPE_WARNING, 500.0, group_size=-3, tracked_objects=[])
    assert res_empty_tracks.group_size == 0
    print("  ✓ Input robustness: Negative distance, invalid zone, None types, and empty tracks handled safely.")


def test_18_extreme_confidence_values():
    print("\n--- TEST 18: Extreme Confidence Values Clamping ---")
    engine = RiskEngine()

    res_under = engine.evaluate(ZONE_TYPE_WARNING, 300.0, confidence=-1.5)
    assert res_under.confidence == 0.0
    assert res_under.contributing_factors["confidence_points"] == 0

    res_over = engine.evaluate(ZONE_TYPE_WARNING, 300.0, confidence=2.5)
    assert res_over.confidence == 1.0
    assert res_over.contributing_factors["confidence_points"] == 10
    print("  ✓ Confidence clamping: -1.5 clamped to 0.0 | 2.5 clamped to 1.0.")


def test_19_score_clamping():
    print("\n--- TEST 19: Score Clamping Boundaries [0, 100] ---")
    engine = RiskEngine()

    # Sum of maximum points across all 8 factors could exceed 100
    res_max = engine.evaluate(
        zone=ZONE_TYPE_PROTECTED,
        distance_to_protected=0.0,
        trend=TREND_APPROACHING,
        group_size=10,
        confidence=1.0,
        persistence_frames=20,
        duration_seconds=60.0,
        recent_alert_count=5,
    )
    raw_sum = sum(res_max.breakdown.values())
    assert raw_sum > 100, f"Raw factor sum should exceed 100, got {raw_sum}"
    assert res_max.score == 100, f"Score must be clamped strictly to 100, got {res_max.score}"

    # Minimum points
    res_min = engine.evaluate(
        zone=ZONE_TYPE_OUTSIDE,
        distance_to_protected=5000.0,
        trend=TREND_RECEDING,
        group_size=0,
        confidence=0.0,
    )
    assert res_min.score >= 0
    print(f"  ✓ Clamping verified: Raw sum={raw_sum} clamped to {res_max.score}/100 | Min score={res_min.score}.")


def test_20_configuration_overrides():
    print("\n--- TEST 20: Configuration Overrides ---")
    # Custom configuration with adjusted thresholds and disabled herd/movement
    custom_engine = RiskEngine(
        low_max=15,
        medium_max=35,
        high_max=60,
        confirmation_frames=3,
        herd_enabled=False,
        movement_enabled=False,
    )

    assessment = custom_engine.evaluate(
        zone=ZONE_TYPE_WARNING,
        distance_to_protected=400.0,
        trend=TREND_APPROACHING,
        group_size=5,  # Herd points disabled
        confidence=0.90,
        persistence_frames=3,  # Conforms to custom 3-frame threshold
    )

    # Movement points disabled
    assert assessment.contributing_factors["trend_points"] == 0
    # Herd points disabled -> solitary points only
    assert assessment.contributing_factors["group_points"] == 2
    # Custom threshold 60 means score >= 61 is CRITICAL
    if assessment.score > 60:
        assert assessment.level == RISK_CRITICAL
    print("  ✓ Configuration overrides: Herd and movement scoring toggled, custom thresholds respected.")


def run_all_tests():
    print("=" * 70)
    print("🐘 RUNNING PROJECT ZOGAN PHASE 7 THREAT ASSESSMENT TEST SUITE")
    print("=" * 70)

    test_1_low_risk_detection()
    test_2_medium_risk_detection()
    test_3_high_risk_detection()
    test_4_critical_risk_detection()
    test_5_single_frame_false_positive()
    test_6_persistent_detection()
    test_7_elephant_leaving_zone()
    test_8_elephant_entering_zone()
    test_9_elephant_approaching_boundary()
    test_10_stationary_elephant()
    test_11_multiple_elephants()
    test_12_risk_decreasing_after_evidence_disappears()
    test_13_event_resolution()
    test_14_alert_cooldown()
    test_15_alert_escalation()
    test_16_deterministic_scoring()
    test_17_invalid_missing_inputs()
    test_18_extreme_confidence_values()
    test_19_score_clamping()
    test_20_configuration_overrides()

    print("\n" + "=" * 70)
    print("🎉 ALL 20 PHASE 7 THREAT ASSESSMENT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
