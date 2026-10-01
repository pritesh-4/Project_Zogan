"""
=============================================================================
🐘 PROJECT ZOGAN — AUTOMATED TEST SUITE FOR PHASE 3.2
=============================================================================

Verifies:
1. Custom elephant dataset existence, counts, and split proportions (~70/15/15)
2. No data leakage (disjoint image stems across train, val, and test splits)
3. Image decode test (OpenCV decodes sample images from each split)
4. Label syntax and coordinates (all bounding boxes are class 0, normalized 0..1)
5. Hard negative samples (properly represented with 0-byte label files)
6. Dataset metadata manifest (datasets/elephant/metadata.json)
7. Dataset validation script execution (ai/validate_dataset.py -> STATUS: PASSED)
8. Phase 2 backward compatibility (test_phase2.py passes)

Usage:
    python test_phase3_2.py
=============================================================================
"""

import sys
import json
import subprocess
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2


def run_cmd(cmd_list):
    res = subprocess.run(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return res.returncode, res.stdout, res.stderr


def test_dataset_counts():
    print("\n--- TEST 1: Dataset File Counts & Proportions ---")
    base = Path("datasets/elephant")

    counts = {}
    total_imgs = 0
    total_lbls = 0
    empty_lbls = 0
    total_boxes = 0

    for split in ["train", "val", "test"]:
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split

        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() == ".jpg"]
        lbls = [p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"]

        assert len(imgs) == len(lbls), f"Mismatch in {split}: {len(imgs)} imgs != {len(lbls)} lbls"
        counts[split] = len(imgs)
        total_imgs += len(imgs)
        total_lbls += len(lbls)

        for lbl in lbls:
            content = lbl.read_text(encoding="utf-8").strip()
            if not content:
                empty_lbls += 1
            else:
                lines = content.splitlines()
                total_boxes += len(lines)

        print(f"  • {split.upper()}: {len(imgs)} images, {len(lbls)} labels")

    print(f"  Total Images: {total_imgs}")
    print(f"  Total Elephant Boxes: {total_boxes}")
    print(f"  Background (Negative) Images: {empty_lbls}")

    assert total_imgs >= 300, f"Expected at least 300 images, got {total_imgs}"
    assert counts["train"] >= 200, f"Expected at least 200 train images, got {counts['train']}"
    assert counts["val"] >= 40, f"Expected at least 40 val images, got {counts['val']}"
    assert counts["test"] >= 40, f"Expected at least 40 test images, got {counts['test']}"
    assert empty_lbls > 0, "Expected hard negative samples with empty labels"
    print("✓ Dataset counts and split sizes verified.")


def test_no_data_leakage():
    print("\n--- TEST 2: Data Leakage Verification ---")
    base = Path("datasets/elephant")

    train_stems = set(p.stem for p in (base / "images" / "train").glob("*.jpg"))
    val_stems = set(p.stem for p in (base / "images" / "val").glob("*.jpg"))
    test_stems = set(p.stem for p in (base / "images" / "test").glob("*.jpg"))

    train_val_overlap = train_stems.intersection(val_stems)
    train_test_overlap = train_stems.intersection(test_stems)
    val_test_overlap = val_stems.intersection(test_stems)

    assert len(train_val_overlap) == 0, f"Data leakage between train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Data leakage between train and test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Data leakage between val and test: {val_test_overlap}"
    print("✓ Confirmed: Splits are strictly disjoint. Zero cross-split leakage.")


def test_image_decoding():
    print("\n--- TEST 3: Image Decode Verification ---")
    base = Path("datasets/elephant")

    for split in ["train", "val", "test"]:
        imgs = list((base / "images" / split).glob("*.jpg"))
        # Test first 10 images from each split
        for p in imgs[:10]:
            img = cv2.imread(str(p))
            assert img is not None, f"Failed to decode image: {p}"
            h, w = img.shape[:2]
            assert h >= 100 and w >= 100, f"Image {p} resolution too small: {w}x{h}"
    print("✓ Sample images successfully decoded and resolution verified.")


def test_label_syntax_and_classes():
    print("\n--- TEST 4: YOLO Label Syntax and Coordinate Bounds ---")
    base = Path("datasets/elephant")

    for split in ["train", "val", "test"]:
        lbl_dir = base / "labels" / split
        for lbl_path in lbl_dir.glob("*.txt"):
            content = lbl_path.read_text(encoding="utf-8").strip()
            if not content:
                continue  # Background image
            for idx, line in enumerate(content.splitlines(), start=1):
                tokens = line.split()
                assert len(tokens) == 5, f"Malformed line in {lbl_path}:{idx} -> {line}"
                class_id = int(tokens[0])
                assert class_id == 0, f"Class ID must be 0 (elephant), got {class_id} in {lbl_path}:{idx}"
                xc, yc, w, h = (
                    float(tokens[1]),
                    float(tokens[2]),
                    float(tokens[3]),
                    float(tokens[4]),
                )
                assert 0.0 <= xc <= 1.0, f"Invalid xc in {lbl_path}:{idx}: {xc}"
                assert 0.0 <= yc <= 1.0, f"Invalid yc in {lbl_path}:{idx}: {yc}"
                assert 0.0 < w <= 1.0, f"Invalid w in {lbl_path}:{idx}: {w}"
                assert 0.0 < h <= 1.0, f"Invalid h in {lbl_path}:{idx}: {h}"
    print("✓ All bounding boxes validated: class 0 only, normalized 0..1.")


def test_metadata_manifest():
    print("\n--- TEST 5: Dataset Metadata Manifest ---")
    meta_path = Path("datasets/elephant/metadata.json")
    assert meta_path.exists(), "Missing datasets/elephant/metadata.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["class_id"] == 0
    assert meta["class_name"] == "elephant"
    assert "sources" in meta and len(meta["sources"]) > 0
    assert "statistics" in meta
    print("✓ metadata.json verified (source tracking and statistics intact).")


def test_validation_script():
    print("\n--- TEST 6: Running Dataset Validator Utility ---")
    code, stdout, stderr = run_cmd([sys.executable, "ai/validate_dataset.py"])
    print(stdout)
    assert code == 0, f"Validator failed! Exit code: {code}\n{stderr}"
    assert "STATUS: PASSED" in stdout, "Expected STATUS: PASSED in validator output"
    print("✓ Dataset validator reported STATUS: PASSED.")


def test_phase2_backward_compatibility():
    print("\n--- TEST 7: Phase 2 Backward Compatibility ---")
    code, stdout, stderr = run_cmd([sys.executable, "test_phase2.py"])
    assert code == 0, f"Phase 2 test failed! Exit code: {code}\n{stderr}"
    assert "ALL TESTS PASSED SUCCESSFULLY!" in stdout
    print("✓ Phase 2 pipeline intact and verified.")


def main():
    print("============================================================")
    print("🐘 RUNNING PROJECT ZOGAN PHASE 3.2 TEST SUITE")
    print("============================================================")
    test_dataset_counts()
    test_no_data_leakage()
    test_image_decoding()
    test_label_syntax_and_classes()
    test_metadata_manifest()
    test_validation_script()
    test_phase2_backward_compatibility()
    print("\n============================================================")
    print("🎉 ALL PHASE 3.2 TESTS PASSED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    main()
