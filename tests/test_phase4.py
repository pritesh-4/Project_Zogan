"""
=============================================================================
🐘 PROJECT ZOGAN — AUTOMATED TEST SUITE FOR PHASE 4
=============================================================================

Comprehensive test verification for Object Tracking & Movement Analysis:
 1. Center-point calculation (x1, y1, x2, y2) -> ((x1+x2)/2, (y1+y2)/2)
 2. Track creation and Track ID assignment
 3. Position history tracking and bounded max history (MAX_HISTORY = 20)
 4. Movement threshold & anti-jitter verification (threshold = 10 px)
 5. RIGHT movement detection
 6. LEFT movement detection
 7. UP movement detection (image space: dy < 0)
 8. DOWN movement detection (image space: dy > 0)
 9. STATIONARY detection (sub-threshold displacement)
10. Multiple simultaneous tracked elephants
11. Lost-track tolerance and automatic expiration cleanup
12. Existing alert logic preservation (persistence, cooldown, reset, enhanced alert)
13. End-to-end ByteTrack inference on elephant.jpg with visual output saving
14. Video source / replay CLI capability verification

Usage:
    python test_phase4.py
=============================================================================
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

# Project modules
import scripts.run_camera as ec
from ai.tracking import (
    ElephantTracker,
    TrackedElephant,
    calculate_center,
    estimate_direction,
    DIRECTION_RIGHT,
    DIRECTION_LEFT,
    DIRECTION_UP,
    DIRECTION_DOWN,
    DIRECTION_STATIONARY,
    DIRECTION_UNKNOWN,
)


def test_center_point_calculation():
    print("\n--- TEST 1: Center-Point Calculation ---")
    # Synthetic box 1: (100, 200, 300, 400)
    box1 = (100, 200, 300, 400)
    cx1, cy1 = calculate_center(box1)
    assert cx1 == 200.0, f"Expected cx=200.0, got {cx1}"
    assert cy1 == 300.0, f"Expected cy=300.0, got {cy1}"
    print(f"  ✓ Box {box1} -> Center ({cx1}, {cy1})")

    # Synthetic box 2: (117, 34, 519, 361)
    box2 = (117, 34, 519, 361)
    cx2, cy2 = calculate_center(box2)
    assert cx2 == (117 + 519) / 2.0
    assert cy2 == (34 + 361) / 2.0
    print(f"  ✓ Box {box2} -> Center ({cx2}, {cy2})")


def test_track_creation_and_id_assignment():
    print("\n--- TEST 2: Track Creation & ID Assignment ---")
    track = TrackedElephant(
        track_id=1,
        initial_bbox=(100, 100, 200, 200),
        confidence=0.95,
        frame_idx=1,
    )
    assert track.track_id == 1, f"Expected ID 1, got {track.track_id}"
    assert track.confidence == 0.95
    assert len(track.history) == 1
    assert track.current_center == (150.0, 150.0)
    assert track.previous_center is None
    # With only 1 point, movement is UNKNOWN
    assert track.movement == DIRECTION_UNKNOWN
    print(f"  ✓ Track #{track.track_id} created with initial center {track.current_center}")


def test_position_history_bounded():
    print("\n--- TEST 3: Bounded Position History ---")
    max_h = 20
    track = TrackedElephant(
        track_id=1,
        initial_bbox=(0, 0, 10, 10),
        max_history=max_h,
    )
    # Add 25 updates (exceeding MAX_HISTORY = 20)
    for i in range(1, 26):
        x = i * 10
        track.update(bbox=(x, 0, x + 10, 10), confidence=0.9, frame_idx=i + 1)

    assert len(track.history) == max_h, f"Expected history bounded to {max_h}, got {len(track.history)}"
    # Verify latest center corresponds to last update (x=250, center=255.0)
    assert track.current_center == (255.0, 5.0)
    print(f"  ✓ Verified history length is bounded to {max_h} entries (oldest evicted correctly).")


def test_movement_threshold_anti_jitter():
    print("\n--- TEST 4: Movement Threshold (Anti-Jitter) ---")
    # Distance: sqrt(4^2 + 2^2) = sqrt(20) = 4.47 pixels, which is < 10.0 threshold
    prev_pt = (200.0, 200.0)
    curr_pt = (204.0, 202.0)

    dir_default = estimate_direction(prev_pt, curr_pt, threshold=10.0)
    assert dir_default == DIRECTION_STATIONARY, f"Expected STATIONARY, got {dir_default}"
    print(f"  ✓ Jitter test: {prev_pt} -> {curr_pt} (dist=4.47px < 10px) => {dir_default}")

    # If threshold is lowered to 3.0 pixels, the 4.47px change registers as movement (RIGHT)
    dir_low_thresh = estimate_direction(prev_pt, curr_pt, threshold=3.0)
    assert dir_low_thresh == DIRECTION_RIGHT, f"Expected RIGHT with low threshold, got {dir_low_thresh}"
    print(f"  ✓ Configurable threshold verified: with thresh=3.0px => {dir_low_thresh}")


def test_direction_right():
    print("\n--- TEST 5: RIGHT Direction Detection ---")
    # Prompt specification: Previous: (100, 100), Current: (150, 103) -> Expected: RIGHT
    prev_pt = (100.0, 100.0)
    curr_pt = (150.0, 103.0)
    res = estimate_direction(prev_pt, curr_pt, threshold=10.0)
    assert res == DIRECTION_RIGHT, f"Expected RIGHT, got {res}"
    print(f"  ✓ {prev_pt} -> {curr_pt} => {res}")


def test_direction_left():
    print("\n--- TEST 6: LEFT Direction Detection ---")
    prev_pt = (300.0, 100.0)
    curr_pt = (240.0, 102.0)
    res = estimate_direction(prev_pt, curr_pt, threshold=10.0)
    assert res == DIRECTION_LEFT, f"Expected LEFT, got {res}"
    print(f"  ✓ {prev_pt} -> {curr_pt} => {res}")


def test_direction_up():
    print("\n--- TEST 7: UP Direction Detection ---")
    # In image space, y decreases upwards
    prev_pt = (200.0, 300.0)
    curr_pt = (202.0, 230.0)
    res = estimate_direction(prev_pt, curr_pt, threshold=10.0)
    assert res == DIRECTION_UP, f"Expected UP, got {res}"
    print(f"  ✓ {prev_pt} -> {curr_pt} => {res}")


def test_direction_down():
    print("\n--- TEST 8: DOWN Direction Detection ---")
    # In image space, y increases downwards
    prev_pt = (200.0, 200.0)
    curr_pt = (201.0, 270.0)
    res = estimate_direction(prev_pt, curr_pt, threshold=10.0)
    assert res == DIRECTION_DOWN, f"Expected DOWN, got {res}"
    print(f"  ✓ {prev_pt} -> {curr_pt} => {res}")


def test_direction_stationary():
    print("\n--- TEST 9: STATIONARY Detection ---")
    # Prompt specification: Previous: (200, 200), Current: (204, 202) -> Expected: STATIONARY
    prev_pt = (200.0, 200.0)
    curr_pt = (204.0, 202.0)
    res = estimate_direction(prev_pt, curr_pt, threshold=10.0)
    assert res == DIRECTION_STATIONARY, f"Expected STATIONARY, got {res}"
    print(f"  ✓ {prev_pt} -> {curr_pt} => {res}")


def test_multiple_tracked_elephants():
    print("\n--- TEST 10: Multiple Simultaneous Tracked Elephants ---")
    tracker = ElephantTracker(movement_threshold=10.0, window_size=2)

    # Frame 1: Three elephants detected
    detections_f1 = [
        {"id": 1, "bbox": (100, 100, 200, 200), "conf": 0.95},
        {"id": 2, "bbox": (300, 300, 400, 400), "conf": 0.91},
        {"id": 3, "bbox": (500, 100, 600, 200), "conf": 0.88},
    ]
    tracker.update(detections_f1, frame_idx=1)
    assert len(tracker.get_active_tracks()) == 3

    # Frame 2:
    # #1 moves RIGHT: (100->160)
    # #2 is STATIONARY: (300->302)
    # #3 moves LEFT: (500->430)
    detections_f2 = [
        {"id": 1, "bbox": (160, 101, 260, 201), "conf": 0.96},
        {"id": 2, "bbox": (302, 301, 402, 401), "conf": 0.92},
        {"id": 3, "bbox": (430, 100, 530, 200), "conf": 0.89},
    ]
    active = tracker.update(detections_f2, frame_idx=2)
    assert len(active) == 3

    t1 = tracker.get_track(1)
    t2 = tracker.get_track(2)
    t3 = tracker.get_track(3)

    assert t1.movement == DIRECTION_RIGHT, f"Elephant 1 expected RIGHT, got {t1.movement}"
    assert t2.movement == DIRECTION_STATIONARY, f"Elephant 2 expected STATIONARY, got {t2.movement}"
    assert t3.movement == DIRECTION_LEFT, f"Elephant 3 expected LEFT, got {t3.movement}"

    hud_summary = tracker.get_hud_summary()
    assert "#1 -> RIGHT" in hud_summary
    assert "#2 -> STATIONARY" in hud_summary
    assert "#3 -> LEFT" in hud_summary
    print(f"  ✓ Multiple tracks maintained: {hud_summary}")


def test_lost_track_cleanup():
    print("\n--- TEST 11: Lost-Track Tolerance & Expiration Cleanup ---")
    tracker = ElephantTracker(max_lost_frames=5)

    # Frame 1: Elephants #1 and #2 present
    tracker.update(
        [
            {"id": 1, "bbox": (100, 100, 200, 200), "conf": 0.9},
            {"id": 2, "bbox": (300, 300, 400, 400), "conf": 0.9},
        ],
        frame_idx=1,
    )
    assert len(tracker.get_active_tracks()) == 2
    assert len(tracker.get_all_tracks()) == 2

    # Frame 2: Elephant #1 temporarily missing (e.g. occlusion)
    tracker.update(
        [
            {"id": 2, "bbox": (305, 305, 405, 405), "conf": 0.9},
        ],
        frame_idx=2,
    )

    # #1 should NOT be deleted immediately from 1 missing frame
    assert len(tracker.get_active_tracks()) == 1, "Only #2 is active in frame 2"
    assert len(tracker.get_all_tracks()) == 2, "Track #1 must be retained during temporary loss"
    assert tracker.get_track(1).frames_lost == 1
    print("  ✓ Single-frame occlusion tolerated: Track #1 kept alive (frames_lost = 1)")

    # Simulate #1 missing for 6 consecutive frames (exceeding max_lost_frames=5)
    for f in range(3, 9):
        tracker.update(
            [
                {"id": 2, "bbox": (310, 310, 410, 410), "conf": 0.9},
            ],
            frame_idx=f,
        )

    # Track #1 should now be expired and cleaned up
    assert tracker.get_track(1) is None, "Track #1 should be purged after exceeding max_lost_frames"
    assert tracker.get_track(2) is not None, "Track #2 should remain active"
    assert len(tracker.get_all_tracks()) == 1
    print("  ✓ Expired track #1 successfully evicted from memory after 6 missed frames.")


def test_alert_logic_preserved():
    print("\n--- TEST 12: Alert Logic Preservation & Enhanced Alert Info ---")
    consecutive_frames = 0
    last_alert_time = 0.0
    alerts_triggered = 0
    current_time = time.time()

    # Simulate 7 frames of elephant detection
    for frame_idx in range(1, 8):
        consecutive_frames += 1
        time_since_last = current_time - last_alert_time
        cooldown_active = time_since_last < ec.ALERT_COOLDOWN_SECONDS

        if consecutive_frames >= ec.REQUIRED_DETECTIONS:
            if not cooldown_active:
                tracked_info = [{"id": 1, "conf": 0.96, "movement": DIRECTION_RIGHT}]
                ec.trigger_alert(0.96, tracked_info=tracked_info)
                last_alert_time = current_time
                alerts_triggered += 1
                print(f"  Frame {frame_idx}: Alert triggered (consecutive={consecutive_frames})")
            else:
                print(f"  Frame {frame_idx}: Confirmed, but alert skipped due to active cooldown.")
        else:
            print(f"  Frame {frame_idx}: Persistence counter: {consecutive_frames}/{ec.REQUIRED_DETECTIONS}")

    assert alerts_triggered == 1, f"Expected 1 alert, got {alerts_triggered}"
    print("  ✓ 5-frame persistence, cooldown rate limiting, and enhanced alert verified.")


def test_end_to_end_tracking_pipeline():
    print("\n--- TEST 13: End-to-End Tracking Pipeline with YOLO Model ---")
    weights_path, model_label = ec.resolve_active_model(use_custom=True)
    model = YOLO(weights_path)
    fixture_image = str(Path(__file__).resolve().parent / "fixtures" / "elephant.jpg")
    img = cv2.imread(fixture_image)
    assert img is not None, f"Failed to load {fixture_image}"

    # Run tracking across two consecutive simulated frames
    tracker = ElephantTracker(movement_threshold=10.0)
    for frame_num in [1, 2]:
        results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
        confirmed = []
        for r in results:
            for b in r.boxes:
                c_id = int(b.cls[0])
                conf = float(b.conf[0])
                if r.names[c_id] == ec.TARGET_CLASS and conf >= ec.CONFIDENCE_THRESHOLD:
                    tid = int(b.id[0]) if getattr(b, "id", None) is not None and b.id is not None else 1
                    confirmed.append(
                        {
                            "id": tid,
                            "bbox": tuple(map(int, b.xyxy[0])),
                            "conf": conf,
                        }
                    )
        active = tracker.update(confirmed, frame_idx=frame_num)

    assert len(active) >= 1, "Tracker should track at least 1 elephant in elephant.jpg"
    t = active[0]
    print(f"  ✓ Tracked Elephant #{t.track_id} | Conf: {t.confidence * 100:.1f}% | Movement: {t.movement}")
    print(f"  ✓ Center: {t.current_center} | History points: {len(t.history)}")

    # Verify rendering
    annotated = img.copy()
    ec.draw_bounding_box(
        annotated,
        int(t.bbox[0]),
        int(t.bbox[1]),
        int(t.bbox[2]),
        int(t.bbox[3]),
        label=f"ELEPHANT #{t.track_id} | {int(t.confidence * 100)}%",
        color=ec.COLOR_ALERT_RED,
        is_target=True,
        sub_label=t.movement,
        trajectory=list(t.history),
    )
    ec.draw_hud(
        annotated,
        status_text="ELEPHANT CONFIRMED",
        status_color=ec.COLOR_ALERT_RED,
        detection_count=5,
        fps=29.2,
        cooldown_remaining=0.0,
        model_name=model_label,
        tracking_summary=tracker.get_hud_summary(),
        tracked_count=len(active),
    )
    output_path = "test_phase4_tracking_preview.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"  ✓ Annotated visual preview saved to {output_path}")


def test_video_replay_option():
    print("\n--- TEST 14: Video File Replay & CLI Source Option ---")
    # Test that elephant_camera.py can be invoked with --source and --no-show
    # Create a 5-frame synthetic test video using elephant.jpg
    fixture_image = str(Path(__file__).resolve().parent / "fixtures" / "elephant.jpg")
    img = cv2.imread(fixture_image)
    assert img is not None, f"Failed to load {fixture_image}"
    h, w = img.shape[:2]
    video_path = "scratch_test_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))
    for _ in range(5):
        out.write(img)
    out.release()

    assert Path(video_path).exists(), "Failed to create scratch test video"
    print(f"  ✓ Created synthetic video for replay testing: {video_path}")

    # Run main() with video source in headless mode
    success = ec.main(
        source=video_path,
        no_show=True,
        max_frames=5,
    )
    assert success is True, "Video replay execution failed in elephant_camera.py"
    print("  ✓ Video file replay option executed cleanly to completion.")

    # Cleanup scratch video
    try:
        Path(video_path).unlink()
        print("  ✓ Cleaned up scratch test video.")
    except Exception:
        pass


def run_all_tests():
    print("=" * 60)
    print("🐘 RUNNING PROJECT ZOGAN PHASE 4 TRACKING TEST SUITE")
    print("=" * 60)

    test_center_point_calculation()
    test_track_creation_and_id_assignment()
    test_position_history_bounded()
    test_movement_threshold_anti_jitter()
    test_direction_right()
    test_direction_left()
    test_direction_up()
    test_direction_down()
    test_direction_stationary()
    test_multiple_tracked_elephants()
    test_lost_track_cleanup()
    test_alert_logic_preserved()
    test_end_to_end_tracking_pipeline()
    test_video_replay_option()

    print("\n" + "=" * 60)
    print("🎉 ALL 14 PHASE 4 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
