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

import sys
import time
import shutil
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from ultralytics import YOLO

# Import project dataset validator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_dataset import DatasetValidator


# =============================================================================
# ⚙️ CONFIGURATION PARAMETERS (Defaults)
# =============================================================================

MODEL_PATH = "yolo26n.pt"
DATASET_PATH = "datasets/elephant/data.yaml"
EPOCHS = 15  # Transfer learning converges rapidly from pretrained weights
IMAGE_SIZE = 416  # High-throughput resolution for CPU/Edge; use 640 for GPU
BATCH_SIZE = 16  # Appropriate batch size for stability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_DIR = "runs/detect"
EXPERIMENT_NAME = "elephant_v1"


def run_preflight_checks(model_path: Path, data_path: Path) -> bool:
    """Verifies that model weights and dataset YAML exist and pass validation."""
    print("=" * 60)
    print("🐘 PROJECT ZOGAN — PRE-FLIGHT TRAINING CHECKS")
    print("=" * 60)

    # 1. Check pretrained model
    if not model_path.exists():
        print(f"❌ Error: Pretrained model weights not found at: {model_path}")
        print("   Please ensure yolo26n.pt is in the project root directory.")
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
    model_str: str = MODEL_PATH,
    data_str: str = DATASET_PATH,
    epochs: int = EPOCHS,
    imgsz: int = IMAGE_SIZE,
    batch: int = BATCH_SIZE,
    device: str = DEVICE,
    project_dir: str = PROJECT_DIR,
    exp_name: str = EXPERIMENT_NAME,
) -> Path:
    """Runs the YOLO fine-tuning pipeline."""
    model_path = Path(model_str)
    data_path = Path(data_str)

    if not run_preflight_checks(model_path, data_path):
        sys.exit(1)

    print("=" * 60)
    print("🚀 STARTING CUSTOM ELEPHANT MODEL TRAINING")
    print("=" * 60)
    print(f"Base Model:       {model_str}")
    print(f"Dataset YAML:     {data_str}")
    print(f"Epochs:           {epochs}")
    print(f"Image Size:       {imgsz}x{imgsz}")
    print(f"Batch Size:       {batch}")
    print(
        f"Compute Device:   {device.upper()} ({torch.get_num_threads()} CPU threads)"
        if device == "cpu"
        else f"Compute Device:   {device.upper()}"
    )
    print(f"Output Directory: {project_dir}/{exp_name}/")
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
        project=project_dir,
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
    best_weights = Path(project_dir) / exp_name / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"⚠️ Warning: Expected best weights at '{best_weights}', falling back to results dir.")
        best_weights = Path(results.save_dir) / "weights" / "best.pt"

    # Organize weights in models/ convention
    deploy_dir = Path("models") / exp_name
    deploy_dir.mkdir(parents=True, exist_ok=True)
    deploy_best = deploy_dir / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, deploy_best)

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total Training Duration:  {mins}m {secs}s ({t_duration:.1f}s)")
    print(f"Experiment Output Dir:    {project_dir}/{exp_name}/")
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
        default=MODEL_PATH,
        help=f"Pretrained base model weights (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATASET_PATH,
        help=f"Path to dataset data.yaml (default: {DATASET_PATH})",
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
        default=PROJECT_DIR,
        help=f"Output project directory (default: {PROJECT_DIR})",
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
