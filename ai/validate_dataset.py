"""
=============================================================================
🐘 PROJECT ZOGAN — DATASET VALIDATION UTILITY (Phase 3.1)
=============================================================================

This script inspects a YOLO-formatted dataset to verify its integrity:
- Checks directory structure and data.yaml configuration
- Ensures 1:1 correspondence between images and label files
- Validates YOLO label formats, class IDs, and normalized coordinates
- Detects malformed lines, out-of-range boxes, and unreadable files
- Computes basic dataset statistics (box counts, averages, min/max per image)
- Provides clear, actionable error messages for beginners

Usage:
    python ai/validate_dataset.py
    python ai/validate_dataset.py --data datasets/elephant/data.yaml

Exit Codes:
    0 = Validation Passed
    1 = Validation Failed (structural issues, missing labels, or invalid annotations)
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure terminal handles UTF-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed. Run: pip install pyyaml")
    sys.exit(1)


# Supported image extensions for YOLO object detection
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard splits expected in a detection dataset
REQUIRED_SPLITS = ["train", "val", "test"]


class DatasetValidator:
    """Validates YOLO dataset structure, image/label pairs, and annotation syntax."""

    def __init__(self, yaml_path: Path):
        self.yaml_path = yaml_path
        self.errors: List[str] = []
        self.warnings: List[str] = []

        # Statistics trackers
        self.split_image_counts: Dict[str, int] = {s: 0 for s in REQUIRED_SPLITS}
        self.split_label_counts: Dict[str, int] = {s: 0 for s in REQUIRED_SPLITS}
        self.split_box_counts: Dict[str, int] = {s: 0 for s in REQUIRED_SPLITS}
        self.empty_label_files: int = 0
        self.missing_label_files: int = 0
        self.orphaned_label_files: int = 0
        self.invalid_annotations: int = 0
        self.valid_annotations: int = 0
        self.boxes_per_image: List[int] = []

        self.dataset_root: Path = yaml_path.parent
        self.config: Dict[str, Any] = {}

    def log_error(self, message: str) -> None:
        """Records a validation error."""
        self.errors.append(message)

    def log_warning(self, message: str) -> None:
        """Records a validation warning (non-fatal)."""
        self.warnings.append(message)

    def validate_yaml(self) -> bool:
        """Checks if data.yaml exists and contains valid required keys."""
        if not self.yaml_path.exists():
            self.log_error(f"Configuration file not found: {self.yaml_path}")
            return False

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.log_error(f"Failed to parse YAML file '{self.yaml_path}': {e}")
            return False

        if not isinstance(self.config, dict):
            self.log_error(f"YAML content in '{self.yaml_path}' is not a valid dictionary.")
            return False

        # Determine dataset root from YAML 'path' or relative to YAML file
        raw_path = self.config.get("path")
        if raw_path:
            p = Path(raw_path)
            if p.is_absolute():
                self.dataset_root = p
            elif (Path.cwd() / p).exists():
                self.dataset_root = Path.cwd() / p
            elif (self.yaml_path.parent / p).exists():
                self.dataset_root = self.yaml_path.parent / p
            else:
                self.dataset_root = Path.cwd() / p
        else:
            self.dataset_root = self.yaml_path.parent

        # Verify classes config
        nc = self.config.get("nc")
        names = self.config.get("names")

        if nc is None:
            self.log_error("YAML missing required key: 'nc' (number of classes).")
        elif not isinstance(nc, int) or nc < 1:
            self.log_error(f"Invalid 'nc' value in YAML: {nc} (must be integer >= 1).")

        if names is None:
            self.log_error("YAML missing required key: 'names' (class names mapping).")
        elif isinstance(names, dict):
            if 0 not in names and "0" not in names:
                self.log_error("YAML 'names' dictionary must include class 0 mapping.")
        elif isinstance(names, list):
            if len(names) == 0:
                self.log_error("YAML 'names' list cannot be empty.")
        else:
            self.log_error("YAML 'names' must be a dict (e.g., {0: 'elephant'}) or list.")

        return len(self.errors) == 0

    def validate_directories(self) -> bool:
        """Verifies that images and labels subdirectories exist for each split."""
        if not self.dataset_root.exists():
            self.log_error(f"Dataset root directory does not exist: {self.dataset_root}")
            return False

        for split in REQUIRED_SPLITS:
            img_rel = self.config.get(split, f"images/{split}")
            img_dir = self.dataset_root / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
            lbl_dir = self.dataset_root / "labels" / split

            if not img_dir.exists():
                self.log_error(f"Missing images directory for split '{split}': {img_dir}")
            if not lbl_dir.exists():
                self.log_error(f"Missing labels directory for split '{split}': {lbl_dir}")

        return len(self.errors) == 0

    def validate_split(self, split: str) -> None:
        """Validates all images and corresponding labels within a single split."""
        img_rel = self.config.get(split, f"images/{split}")
        img_dir = self.dataset_root / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
        lbl_dir = self.dataset_root / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            return

        # Collect image files (ignore dotfiles like .gitkeep)
        image_files: Dict[str, Path] = {}
        for p in img_dir.iterdir():
            if p.is_file() and not p.name.startswith("."):
                ext = p.suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    stem = p.stem
                    if stem in image_files:
                        self.log_error(
                            f"Duplicate image stem '{stem}' with different extensions in {img_dir} "
                            f"({p.name} and {image_files[stem].name})"
                        )
                    else:
                        image_files[stem] = p
                else:
                    self.log_warning(
                        f"Skipping unsupported image extension: {p.relative_to(self.dataset_root)}"
                    )

        # Collect label files
        label_files: Dict[str, Path] = {}
        for p in lbl_dir.iterdir():
            if p.is_file() and not p.name.startswith("."):
                if p.suffix.lower() == ".txt":
                    label_files[p.stem] = p
                else:
                    self.log_warning(
                        f"Non-text file found in labels folder: {p.relative_to(self.dataset_root)}"
                    )

        self.split_image_counts[split] = len(image_files)
        self.split_label_counts[split] = len(label_files)

        # 1. Check for images missing corresponding labels
        for stem, img_path in image_files.items():
            if stem not in label_files:
                self.missing_label_files += 1
                expected_lbl = lbl_dir / f"{stem}.txt"
                self.log_error(
                    f"Missing label file: '{expected_lbl.relative_to(self.dataset_root)}' "
                    f"for image '{img_path.relative_to(self.dataset_root)}'"
                )

        # 2. Check for orphaned label files (no matching image)
        for stem, lbl_path in label_files.items():
            if stem not in image_files:
                self.orphaned_label_files += 1
                self.log_error(
                    f"Orphaned label file (no matching image): '{lbl_path.relative_to(self.dataset_root)}'"
                )

        # 3. Validate contents of each label file that has a corresponding image
        max_class_id = (self.config.get("nc", 1) - 1)

        for stem, lbl_path in label_files.items():
            if stem not in image_files:
                continue

            rel_lbl = lbl_path.relative_to(self.dataset_root)

            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                self.invalid_annotations += 1
                self.log_error(f"Unreadable label file '{rel_lbl}': {e}")
                continue

            # Strip whitespace and filter empty lines
            clean_lines = [line.strip() for line in lines if line.strip()]

            if len(clean_lines) == 0:
                # Valid negative/background sample (image with no elephants)
                self.empty_label_files += 1
                self.boxes_per_image.append(0)
                continue

            box_count_for_file = 0

            for line_idx, line in enumerate(clean_lines, start=1):
                tokens = line.split()

                # Must contain exactly 5 space-separated values
                if len(tokens) != 5:
                    self.invalid_annotations += 1
                    self.log_error(
                        f"Malformed annotation line {line_idx} in '{rel_lbl}': "
                        f"Expected 5 values (<class_id> <x_center> <y_center> <width> <height>), "
                        f"got {len(tokens)} -> '{line}'"
                    )
                    continue

                # Validate class_id
                try:
                    class_id = int(tokens[0])
                except ValueError:
                    self.invalid_annotations += 1
                    self.log_error(
                        f"Invalid non-integer class_id '{tokens[0]}' in '{rel_lbl}' (line {line_idx})"
                    )
                    continue

                if class_id < 0 or class_id > max_class_id:
                    self.invalid_annotations += 1
                    self.log_error(
                        f"Class ID {class_id} in '{rel_lbl}' (line {line_idx}) out of range (expected 0..{max_class_id})"
                    )
                    continue

                # Validate normalized bounding box coordinates
                try:
                    xc = float(tokens[1])
                    yc = float(tokens[2])
                    w = float(tokens[3])
                    h = float(tokens[4])
                except ValueError:
                    self.invalid_annotations += 1
                    self.log_error(
                        f"Non-numeric coordinates in '{rel_lbl}' (line {line_idx}) -> '{line}'"
                    )
                    continue

                # Check coordinates within valid range [0, 1]
                coord_errors = []
                if not (0.0 <= xc <= 1.0):
                    coord_errors.append(f"x_center={xc:.4f} not in [0.0, 1.0]")
                if not (0.0 <= yc <= 1.0):
                    coord_errors.append(f"y_center={yc:.4f} not in [0.0, 1.0]")
                if not (0.0 < w <= 1.0):
                    coord_errors.append(f"width={w:.4f} must be > 0.0 and <= 1.0")
                if not (0.0 < h <= 1.0):
                    coord_errors.append(f"height={h:.4f} must be > 0.0 and <= 1.0")

                if coord_errors:
                    self.invalid_annotations += 1
                    self.log_error(
                        f"Invalid coordinates in '{rel_lbl}' (line {line_idx}): {', '.join(coord_errors)}"
                    )
                    continue

                # Check bounding box bounds
                x1 = xc - w / 2.0
                x2 = xc + w / 2.0
                y1 = yc - h / 2.0
                y2 = yc + h / 2.0

                if x1 < -0.05 or x2 > 1.05 or y1 < -0.05 or y2 > 1.05:
                    self.log_warning(
                        f"Bounding box extends significantly outside image in '{rel_lbl}' (line {line_idx}): "
                        f"[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]"
                    )

                self.valid_annotations += 1
                box_count_for_file += 1
                self.split_box_counts[split] += 1

            self.boxes_per_image.append(box_count_for_file)

    def run(self) -> bool:
        """Executes full validation pipeline and prints formatted report."""
        print("=" * 60)
        print("PROJECT ZOGAN — DATASET VALIDATION (Phase 3.1)")
        print("=" * 60)
        print(f"Config File:  {self.yaml_path}")

        # Step 1: Validate YAML syntax and configuration
        if not self.validate_yaml():
            self.print_report()
            return False

        print(f"Dataset Root: {self.dataset_root}")
        print(f"Classes:      {self.config.get('nc')} {self.config.get('names')}")
        print("-" * 60)

        # Step 2: Validate directory structure
        if not self.validate_directories():
            self.print_report()
            return False

        # Step 3: Validate each split
        for split in REQUIRED_SPLITS:
            self.validate_split(split)

        # Step 4: Output results
        self.print_report()
        return len(self.errors) == 0

    def print_report(self) -> None:
        """Outputs beginner-friendly summary and statistics."""
        total_images = sum(self.split_image_counts.values())
        total_boxes = sum(self.split_box_counts.values())

        annotated_images_boxes = [b for b in self.boxes_per_image if b > 0]
        avg_boxes = (
            sum(annotated_images_boxes) / len(annotated_images_boxes)
            if annotated_images_boxes
            else 0.0
        )
        min_boxes = min(annotated_images_boxes) if annotated_images_boxes else 0
        max_boxes = max(annotated_images_boxes) if annotated_images_boxes else 0

        # Print warnings if any
        if self.warnings:
            print("\nWARNINGS:")
            for w in self.warnings[:15]:
                print(f"  [!] {w}")
            if len(self.warnings) > 15:
                print(f"  ... and {len(self.warnings) - 15} more warnings.")

        # Print errors if any
        if self.errors:
            print("\nERRORS FOUND:")
            for err in self.errors[:25]:
                print(f"  [X] {err}")
            if len(self.errors) > 25:
                print(f"  ... and {len(self.errors) - 25} more errors.")

        # Print summary format requested
        print("\n" + "=" * 60)
        print("DATASET VALIDATION")
        print("-" * 60)
        print(f"Train images:       {self.split_image_counts['train']}")
        print(f"Validation images:  {self.split_image_counts['val']}")
        print(f"Test images:        {self.split_image_counts['test']}")
        print(f"Total images:       {total_images}")
        print("")
        print(f"Valid annotations:  {self.valid_annotations}")
        print(f"Background images:  {self.empty_label_files} (empty labels with 0 boxes)")
        print(f"Missing labels:     {self.missing_label_files}")
        print(f"Orphaned labels:    {self.orphaned_label_files}")
        print(f"Invalid labels:     {self.invalid_annotations}")

        if total_images > 0 and self.valid_annotations > 0:
            print("\nDATASET STATISTICS:")
            print(f"Total elephant boxes:              {total_boxes}")
            print(f"Average elephants/annotated image: {avg_boxes:.2f}")
            print(f"Min / Max elephants per image:     {min_boxes} / {max_boxes}")
        elif total_images == 0:
            print("\nNOTE: Dataset directory structure and YAML are valid.")
            print("The dataset currently contains 0 images.")
            print("Ready for image collection and annotation in Phase 3.2.")

        print("-" * 60)
        if len(self.errors) == 0:
            print("STATUS: PASSED")
        else:
            print("STATUS: FAILED")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate a YOLO object detection dataset structure and annotations."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/elephant/data.yaml",
        help="Path to dataset data.yaml file (default: datasets/elephant/data.yaml)",
    )
    args = parser.parse_args()

    yaml_path = Path(args.data)
    if not yaml_path.is_absolute():
        yaml_path = Path.cwd() / yaml_path

    validator = DatasetValidator(yaml_path)
    passed = validator.run()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
