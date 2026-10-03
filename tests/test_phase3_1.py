"""
=============================================================================
🐘 PROJECT ZOGAN — AUTOMATED TEST SUITE FOR PHASE 3.1
=============================================================================

Verifies:
1. Dataset directory structure (images/train, val, test and labels/train, val, test)
2. data.yaml structure and schema (nc: 1, class 0 = elephant)
3. datasets/elephant/README.md documentation presence
4. Dataset validator on the clean repository (STATUS: PASSED, exit code 0)
5. Dataset validator error detection capabilities (mocked cases):
   - Missing label detection
   - Orphaned label detection
   - Malformed annotation line detection
   - Out-of-bounds coordinate detection
   - Invalid class ID detection
   - Valid background image handling (empty label)
6. Dataset visualizer execution on sample image + annotation
7. Clean workspace teardown after mock testing

Usage:
    python test_phase3_1.py
=============================================================================
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd_list, cwd=None):
    """Executes a subprocess command and returns (returncode, stdout, stderr)."""
    res = subprocess.run(
        cmd_list,
        cwd=cwd or REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return res.returncode, res.stdout, res.stderr


def test_directory_structure():
    print("\n--- TEST 1: Dataset Directory Structure ---")
    base = Path("datasets/elephant")
    assert base.exists(), f"Missing dataset base directory: {base}"

    expected_dirs = [
        base / "images" / "train",
        base / "images" / "val",
        base / "images" / "test",
        base / "labels" / "train",
        base / "labels" / "val",
        base / "labels" / "test",
    ]
    for d in expected_dirs:
        assert d.exists() and d.is_dir(), f"Missing required directory: {d}"
        print(f"  ✓ Found directory: {d}")
    print("✓ All dataset directories exist.")


def test_data_yaml():
    print("\n--- TEST 2: data.yaml Configuration ---")
    yaml_path = Path("datasets/elephant/data.yaml")
    assert yaml_path.exists(), f"Missing {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "data.yaml must be a dictionary"
    assert "train" in config, "data.yaml missing 'train'"
    assert "val" in config, "data.yaml missing 'val'"
    assert "test" in config, "data.yaml missing 'test'"
    assert config.get("nc") == 1, f"Expected nc: 1, got {config.get('nc')}"

    names = config.get("names")
    if isinstance(names, dict):
        assert names.get(0) == "elephant" or names.get("0") == "elephant", f"Expected class 0: elephant, got {names}"
    elif isinstance(names, list):
        assert len(names) == 1 and names[0] == "elephant", f"Expected ['elephant'], got {names}"
    else:
        raise AssertionError(f"Unexpected 'names' type: {type(names)}")

    print(f"  ✓ data.yaml is valid: nc={config['nc']}, names={config['names']}")


def test_readme_docs():
    print("\n--- TEST 3: Dataset Documentation ---")
    readme_path = Path("datasets/elephant/README.md")
    assert readme_path.exists(), f"Missing {readme_path}"

    content = readme_path.read_text(encoding="utf-8")
    assert "elephant" in content.lower(), "README should mention elephant"
    assert "0" in content, "README should explain class 0"
    assert "x_center" in content or "x_center" in content.lower(), "README should explain YOLO format"
    assert "train" in content and "val" in content and "test" in content, "README should explain splits"
    print(f"  ✓ datasets/elephant/README.md verified ({len(content)} characters).")


def test_validator_on_clean_repo():
    print("\n--- TEST 4: Dataset Validator on Clean Dataset ---")
    code, stdout, stderr = run_cmd([sys.executable, "ai/validate_dataset.py"])
    print(stdout)
    assert code == 0, f"Validator failed on clean repo! Exit code: {code}\n{stderr}"
    assert "STATUS: PASSED" in stdout, "Expected STATUS: PASSED in validator output"
    print("✓ Validator passed successfully on clean repository.")


def test_validator_error_cases():
    print("\n--- TEST 5: Validator Error Detection (Mock Scenarios) ---")
    # Create an isolated temporary test directory
    temp_dir = Path(tempfile.mkdtemp(prefix="zogan_dataset_test_"))
    try:
        images_train = temp_dir / "images" / "train"
        labels_train = temp_dir / "labels" / "train"
        images_train.mkdir(parents=True)
        labels_train.mkdir(parents=True)

        for split in ["val", "test"]:
            (temp_dir / "images" / split).mkdir(parents=True)
            (temp_dir / "labels" / split).mkdir(parents=True)

        # Create dummy image (100x100 black square)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        # 1. Valid case
        cv2.imwrite(str(images_train / "valid_001.jpg"), dummy_img)
        (labels_train / "valid_001.txt").write_text("0 0.500 0.500 0.300 0.400\n", encoding="utf-8")

        # 2. Valid background case (empty label file)
        cv2.imwrite(str(images_train / "bg_001.jpg"), dummy_img)
        (labels_train / "bg_001.txt").write_text("", encoding="utf-8")

        mock_yaml = temp_dir / "data.yaml"
        mock_config = {
            "path": str(temp_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": {0: "elephant"},
        }
        with open(mock_yaml, "w", encoding="utf-8") as f:
            yaml.dump(mock_config, f)

        # Check valid mock dataset
        code, stdout, _ = run_cmd([sys.executable, "ai/validate_dataset.py", "--data", str(mock_yaml)])
        assert code == 0 and "STATUS: PASSED" in stdout, "Mock valid dataset failed validation!"
        assert "Total elephant boxes:              1" in stdout or "Total elephant boxes" in stdout
        print("  ✓ Mock valid sample + background image: PASSED")

        # 3. Missing label case
        cv2.imwrite(str(images_train / "no_label.jpg"), dummy_img)
        code, stdout, _ = run_cmd([sys.executable, "ai/validate_dataset.py", "--data", str(mock_yaml)])
        assert code == 1 and "STATUS: FAILED" in stdout, "Validator failed to flag missing label!"
        assert "Missing label file" in stdout
        print("  ✓ Missing label file correctly detected: FAILED as expected")
        (images_train / "no_label.jpg").unlink()

        # 4. Orphaned label case
        (labels_train / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        code, stdout, _ = run_cmd([sys.executable, "ai/validate_dataset.py", "--data", str(mock_yaml)])
        assert code == 1 and "STATUS: FAILED" in stdout, "Validator failed to flag orphaned label!"
        assert "Orphaned label file" in stdout
        print("  ✓ Orphaned label file correctly detected: FAILED as expected")
        (labels_train / "orphan.txt").unlink()

        # 5. Malformed annotation line (4 tokens instead of 5)
        (labels_train / "valid_001.txt").write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
        code, stdout, _ = run_cmd([sys.executable, "ai/validate_dataset.py", "--data", str(mock_yaml)])
        assert code == 1 and "STATUS: FAILED" in stdout, "Validator failed to flag malformed line!"
        assert "Malformed annotation" in stdout
        print("  ✓ Malformed annotation line (4 values) correctly detected: FAILED as expected")

        # 6. Out of bounds coordinate (x_center = 1.45)
        (labels_train / "valid_001.txt").write_text("0 1.45 0.5 0.2 0.2\n", encoding="utf-8")
        code, stdout, _ = run_cmd([sys.executable, "ai/validate_dataset.py", "--data", str(mock_yaml)])
        assert code == 1 and "STATUS: FAILED" in stdout, "Validator failed to flag out-of-bounds coordinate!"
        assert "Invalid coordinates" in stdout
        print("  ✓ Out-of-bounds coordinate correctly detected: FAILED as expected")

        # 7. Invalid class ID (class 2 when nc=1)
        (labels_train / "valid_001.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        code, stdout, _ = run_cmd([sys.executable, "ai/validate_dataset.py", "--data", str(mock_yaml)])
        assert code == 1 and "STATUS: FAILED" in stdout, "Validator failed to flag invalid class ID!"
        assert "Class ID" in stdout and "out of range" in stdout
        print("  ✓ Invalid class ID correctly detected: FAILED as expected")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("✓ All validation error checks verified successfully.")


def test_visualizer():
    print("\n--- TEST 6: Visualizer Execution ---")
    temp_dir = Path(tempfile.mkdtemp(prefix="zogan_vis_test_"))
    try:
        dummy_img = np.zeros((200, 300, 3), dtype=np.uint8)
        img_path = temp_dir / "test_elephant.jpg"
        lbl_path = temp_dir / "test_elephant.txt"
        out_path = temp_dir / "preview.jpg"

        cv2.imwrite(str(img_path), dummy_img)
        lbl_path.write_text("0 0.500 0.500 0.400 0.600\n", encoding="utf-8")

        code, stdout, stderr = run_cmd(
            [
                sys.executable,
                "ai/visualize_dataset.py",
                str(img_path),
                "--save",
                str(out_path),
                "--no-show",
            ]
        )
        assert code == 0, f"Visualizer exited with error: {stderr}\n{stdout}"
        assert out_path.exists(), f"Visualizer did not generate output at {out_path}"
        print(f"  ✓ Visualizer generated annotated image: {out_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("✓ Visualizer test passed.")


def main():
    print("============================================================")
    print("🐘 RUNNING PROJECT ZOGAN PHASE 3.1 TEST SUITE")
    print("============================================================")
    test_directory_structure()
    test_data_yaml()
    test_readme_docs()
    test_validator_on_clean_repo()
    test_validator_error_cases()
    test_visualizer()
    print("\n============================================================")
    print("🎉 ALL PHASE 3.1 TESTS PASSED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    main()
