# 🐘 Project Zogan — Models Directory & Changelog

This directory manages trained, fine-tuned, and deployed model checkpoints for Project Zogan.

```text
models/
├── README.md              # Model changelog, specifications, and deployment guide
└── elephant_v1/
    ├── best.pt            # Deployed fine-tuned custom elephant model weights
    └── README.md          # Experiment-specific documentation
```

---

## 📋 Model Changelog & Specifications

### 1. `elephant_v1` (Primary Custom Model — Research / Prototype)

* **Model Name**: `elephant_v1`
* **Architecture**: YOLO26n (Ultralytics Nano Object Detector)
* **Base Weights**: `yolo26n.pt` (Pretrained COCO transfer learning)
* **Status**: Fine-tuned prototype for evaluation (not certified for unsupervised production deployment)
* **Dataset Version**: Phase 3.2 African Wildlife Dataset (456 images [315 train / 68 val / 73 test], 747 boxes)
* **Training Configuration**:
  * Epochs: 15
  * Resolution: $416 \times 416$
  * Batch Size: 16
  * Optimizer: AdamW
  * Device: 12-core CPU
* **Evaluation Metrics (Held-Out Test Set)**:
  * **Precision**: 83.3%
  * **Recall**: 76.8%
  * **mAP@50**: 88.2%
  * **mAP@50-95**: 72.4%
  * **Inference Latency**: ~21–50 ms/frame
* **Intended Use**: Real-time daytime elephant early-warning detection for webcam and field cameras.
* **Current Limitations**:
  * Trained on African bush elephants; Asian elephants not yet explicitly represented.
  * Daytime RGB only; night/infrared imagery not included.
  * Heavily obscured elephants behind dense foliage show lower confidence.

### 2. `yolo26n.pt` (Baseline Fallback Model)

* **Model Name**: Pretrained YOLO26n (COCO)
* **Target Class**: Elephant (COCO class ID 20)
* **Intended Use**: Baseline benchmarking and fallback verification.

---

## 🔄 How to Switch Models in `elephant_camera.py`

### Option A: Configuration Flag (Code)
In [`elephant_camera.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/elephant_camera.py):
```python
# Set True for custom fine-tuned model (default)
USE_CUSTOM_MODEL = True

# Set False to fall back to general COCO pretrained model
USE_CUSTOM_MODEL = False
```

### Option B: Command-Line Flag (CLI)
```powershell
# Run with custom model (default)
python elephant_camera.py

# Force fallback to pretrained model
python elephant_camera.py --pretrained

# Explicit weights path
python elephant_camera.py --model models/elephant_v1/best.pt
```

---

## 💾 Model Weights & Git Best Practices

* Model weights (`.pt` files) are binary PyTorch archives (~5.3 MB for YOLO nano models).
* For larger models or frequent checkpoints, track via Git LFS:
  ```powershell
  git lfs track "*.pt"
  ```
* Checkpoints in `runs/detect/` should be treated as training artifacts, while release models are curated into `models/<name>/best.pt`.
