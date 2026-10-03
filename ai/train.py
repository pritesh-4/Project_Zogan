"""
=============================================================================
🐘 PROJECT ZOGAN — CUSTOM ELEPHANT MODEL TRAINING (Phase 3.3)
=============================================================================

This script fine-tunes a pretrained YOLO model (yolo26n.pt) on the custom
elephant detection dataset (datasets/elephant/data.yaml) using transfer learning.

Features:
- Auto-detects device (CUDA GPU if available, otherwise multi-core CPU)
- Pre-flight dataset validation check before launching training
- Configurable hyperparameters (epochs, image size, batch size, experiment name)
- Saves checkpoints and training analytics in runs/detect/<experiment_name>/
- Organizes best weights into models/<experiment_name>/ for deployment

Usage:
    python ai/train.py
    python ai/train.py --epochs 20 --imgsz 416 --batch 16
    python ai/train.py --epochs 50 --imgsz 640 --batch 16

=============================================================================
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from ultralytics import YOLO

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.validate_dataset import DatasetValidator

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
# ⚙️ CONFIGURATION PARAMETERS & PATH RESOLUTION
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_model_path(requested_model: str | Path | None = None) -> Path:
    """Resolves base model weights file from cwd, repo root, or models/ directory."""
    if requested_model:
        p = Path(requested_model)
        if p.is_absolute() and p.exists():
            return p
        if (Path.cwd() / p).exists():
            return (Path.cwd() / p).resolve()
        if (REPO_ROOT / p).exists():
            return (REPO_ROOT / p).resolve()
        if (REPO_ROOT / "models" / p.name).exists():
            return (REPO_ROOT / "models" / p.name).resolve()
        return p

    candidates = [
        REPO_ROOT / "models" / "yolo26n.pt",
        Path.cwd() / "models" / "yolo26n.pt",
        Path.cwd() / "yolo26n.pt",
        REPO_ROOT / "yolo26n.pt",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (REPO_ROOT / "models" / "yolo26n.pt").resolve()


def resolve_dataset_path(requested_data: str | Path | None = None) -> Path:
    """Resolves dataset YAML configuration from cwd or repo root."""
    if requested_data:
        p = Path(requested_data)
        if p.is_absolute() and p.exists():
            return p
        if (Path.cwd() / p).exists():
            return (Path.cwd() / p).resolve()
        if (REPO_ROOT / p).exists():
            return (REPO_ROOT / p).resolve()
        return p

    candidates = [
        REPO_ROOT / "datasets" / "elephant" / "data.yaml",
        Path.cwd() / "datasets" / "elephant" / "data.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (REPO_ROOT / "datasets" / "elephant" / "data.yaml").resolve()


DEFAULT_MODEL_STR = "models/yolo26n.pt"
DEFAULT_DATASET_STR = "datasets/elephant/data.yaml"
DEFAULT_PROJECT_DIR = "runs/detect"

MODEL_PATH = str(resolve_model_path())
DATASET_PATH = str(resolve_dataset_path())
EPOCHS = 15  # Transfer learning converges rapidly from pretrained weights
IMAGE_SIZE = 416  # High-throughput resolution for CPU/Edge; use 640 for GPU
BATCH_SIZE = 16  # Appropriate batch size for stability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_DIR = DEFAULT_PROJECT_DIR
EXPERIMENT_NAME = "elephant_v1"


def run_preflight_checks(model_path: Path, data_path: Path) -> bool:
    """Verifies that model weights and dataset YAML exist and pass validation."""
    if Path.cwd() != REPO_ROOT:
        os.chdir(REPO_ROOT)

    print("=" * 60)
    print("🐘 PROJECT ZOGAN — PRE-FLIGHT TRAINING CHECKS")
    print("=" * 60)

    # 1. Check pretrained model
    if not model_path.exists():
        print(f"❌ Error: Pretrained model weights not found at: {model_path}")
        print("   Please ensure yolo26n.pt is in the models/ directory.")
        return False
    print(f"[OK] Pretrained base model found: {model_path}")

    # 2. Check dataset configuration
    if not data_path.exists():
        print(f"❌ Error: Dataset configuration file not found at: {data_path}")
        return False
    print(f"[OK] Dataset configuration found: {data_path}")

    # 3. Run dataset validator
    print("\nRunning dataset integrity check...")
    validator = DatasetValidator(data_path)
    passed = validator.run()
    if not passed:
        print("\n❌ Error: Dataset validation failed! Fix dataset errors before training.")
        return False
    print("[OK] Dataset validation passed successfully.\n")
    return True


def train_custom_model(
    model_str: str = DEFAULT_MODEL_STR,
    data_str: str = DEFAULT_DATASET_STR,
    epochs: int = EPOCHS,
    imgsz: int = IMAGE_SIZE,
    batch: int = BATCH_SIZE,
    device: str = DEVICE,
    project_dir: str = DEFAULT_PROJECT_DIR,
    exp_name: str = EXPERIMENT_NAME,
) -> Path:
    """Runs the YOLO fine-tuning pipeline."""
    # Resolve paths before anchoring CWD
    model_path = resolve_model_path(model_str)
    data_path = resolve_dataset_path(data_str)

    # Anchor CWD to repository root for consistent relative paths and YOLO dataset resolution
    if Path.cwd() != REPO_ROOT:
        os.chdir(REPO_ROOT)

    if not run_preflight_checks(model_path, data_path):
        sys.exit(1)

    # Resolve project directory as absolute path to avoid Ultralytics nested prefixing
    if Path(project_dir).is_absolute():
        resolved_project_dir = Path(project_dir)
    else:
        resolved_project_dir = (REPO_ROOT / project_dir).resolve()

    print("=" * 60)
    print("🚀 STARTING CUSTOM ELEPHANT MODEL TRAINING")
    print("=" * 60)
    print(f"Base Model:       {model_path}")
    print(f"Dataset YAML:     {data_path}")
    print(f"Epochs:           {epochs}")
    print(f"Image Size:       {imgsz}x{imgsz}")
    print(f"Batch Size:       {batch}")
    print(
        f"Compute Device:   {device.upper()} ({torch.get_num_threads()} CPU threads)"
        if device == "cpu"
        else f"Compute Device:   {device.upper()}"
    )
    print(f"Output Directory: {resolved_project_dir}/{exp_name}/")
    print("=" * 60 + "\n")

    t_start = time.time()

    # Load pretrained YOLO model
    model = YOLO(str(model_path))

    # Train model on custom elephant dataset
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(resolved_project_dir),
        name=exp_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    t_duration = time.time() - t_start
    mins = int(t_duration // 60)
    secs = int(t_duration % 60)

    # Locate best weights
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (resolved_project_dir / exp_name)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        fallback_weights = resolved_project_dir / exp_name / "weights" / "best.pt"
        if fallback_weights.exists():
            best_weights = fallback_weights
        else:
            print(f"⚠️ Warning: Best weights not found at expected location: {best_weights}")

    # Organize weights in models/ convention
    deploy_dir = REPO_ROOT / "models" / exp_name
    deploy_dir.mkdir(parents=True, exist_ok=True)
    deploy_best = deploy_dir / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, deploy_best)

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total Training Duration:  {mins}m {secs}s ({t_duration:.1f}s)")
    print(f"Experiment Output Dir:    {save_dir}/")
    print(f"Best Model Weights:       {best_weights}")
    print(f"Deployment Model Weights: {deploy_best}")
    print("-" * 60)
    print("Next step: Evaluate the model on held-out test data:")
    print(f"    python ai/evaluate.py --weights {deploy_best}")
    print("=" * 60)

    return best_weights


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on custom elephant detection dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_STR,
        help=f"Pretrained base model weights (default: {DEFAULT_MODEL_STR})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATASET_STR,
        help=f"Path to dataset data.yaml (default: {DEFAULT_DATASET_STR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
        help=f"Input image resolution (default: {IMAGE_SIZE})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device: 'cpu' or 'cuda' (default: {DEVICE})",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT_DIR,
        help=f"Output project directory (default: {DEFAULT_PROJECT_DIR})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=EXPERIMENT_NAME,
        help=f"Experiment run name (default: {EXPERIMENT_NAME})",
    )
    args = parser.parse_args()

    train_custom_model(
        model_str=args.model,
        data_str=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project_dir=args.project,
        exp_name=args.name,
    )


if __name__ == "__main__":
    main()
