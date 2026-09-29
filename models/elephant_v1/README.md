# 🐘 Project Zogan — Models Directory

## Model Checkpoint Convention

This directory stores deployed and versioned model weights for Project Zogan.

```text
models/
└── elephant_v1/
    ├── best.pt        # Optimal checkpoint weights from fine-tuning
    └── README.md      # Model documentation and version log
```

### Experiment Log

| Version | Base Model | Dataset | Epochs | mAP50 | mAP50-95 | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`elephant_v1`** | `yolo26n.pt` | `datasets/elephant` (456 imgs) | 15 | *See evaluation* | *See evaluation* | First custom fine-tuned model for single-class elephant detection. |

---

## 💾 Model Weights & Git Best Practices

* **Model Checkpoints**: YOLO `.pt` files are PyTorch binary archives (~5 MB for nano models).
* For larger models or high-frequency experiments:
  * Store weights in Git LFS (Large File Storage) or release assets:
    `git lfs track "*.pt"`
  * Alternatively, download weights on-demand during deployment or attach to GitHub Releases.
