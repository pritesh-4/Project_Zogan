# 🐘 Project Zogan — Custom Elephant Detection Dataset

> **Phase 3.2: Custom Elephant Dataset Specification, Curation & Statistics**  
> Target: Single-class YOLO Object Detection format (`0: elephant`) with hard negative background samples.

---

## 📊 Current Dataset Overview (Phase 3.2)

The dataset is fully populated, validated, and ready for model training.

### Summary Statistics

| Metric | Count | Proportion |
| :--- | :--- | :--- |
| **Total Images** | **456** | 100.0% |
| **Positive Images (Contains Elephants)** | **382** | 83.8% |
| **Hard Negative Images (Background / No Elephants)** | **74** | 16.2% |
| **Total Elephant Bounding Boxes** | **747** | — |
| **Average Elephants per Annotated Image** | **1.96** | — |
| **Min / Max Elephants in a Single Image** | **1 / 14** | — |

### Split Breakdown

| Split | Positive Images | Negative Images | Total Images | Total Boxes | Split Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 271 | 44 | **315** | 557 | 69.1% |
| **Validation** | 53 | 15 | **68** | 91 | 14.9% |
| **Test** | 58 | 15 | **73** | 99 | 16.0% |
| **Total** | **382** | **74** | **456** | **747** | **100.0%** |

---

## 📖 1. Dataset Purpose

In Phase 1 and Phase 2, Project Zogan used a general pre-trained YOLO model (`yolo26n.pt`) trained on the COCO dataset (80 generic classes). While COCO includes an elephant category, general-purpose models often produce false alarms when viewing large grazing livestock (cows, buffaloes) or dense bush.

This custom dataset focuses exclusively on **Elephant Object Detection**:
* Establishes dedicated ground-truth bounding boxes for wild elephants.
* Introduces hard negative samples (savanna vegetation, buffaloes, rhinos, zebras) with empty labels to teach the model to ignore non-target animals and background noise.
* Prepares the foundation for training a fine-tuned, single-class model in Phase 3.3.

---

## 🏷️ 2. Dataset Sources & Licensing

The initial Phase 3.2 dataset is curated from the official open **Ultralytics African Wildlife Dataset**:

* **Dataset Name**: African Wildlife Detection Dataset
* **Curator**: Ultralytics
* **Source URL**: [https://docs.ultralytics.com/datasets/detect/african-wildlife](https://docs.ultralytics.com/datasets/detect/african-wildlife)
* **Archive URL**: `https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip`
* **License**: GNU Affero General Public License v3.0 (AGPL-3.0)
* **Curation & Preprocessing Applied**:
  1. Filtered all images containing elephants (original class `1`), re-mapping bounding boxes to single class `0` (`elephant`).
  2. Identified non-elephant images (buffaloes, rhinos, zebras, savanna vegetation) and sampled 74 representative scenes to serve as **hard negative background images** (represented by 0-byte `.txt` label files).
  3. Verified all image decodes using OpenCV and validated all bounding box coordinates to ensure strict $0.0 \le x, y \le 1.0$ and $w, h > 0$.
  4. Standardized file naming to clean identifiers (`elephant_{split}_{id}.jpg` and `background_{split}_{id}.jpg`).
  5. Full traceability is documented in [`metadata.json`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/datasets/elephant/metadata.json).

---

## 📁 3. Folder Structure

The dataset strictly follows standard Ultralytics YOLO layout:

```text
datasets/
└── elephant/
    ├── data.yaml            # YOLO dataset configuration
    ├── README.md            # This documentation guide
    ├── metadata.json        # Manifest tracking original filenames and box counts
    │
    ├── images/              # Source images (.jpg)
    │   ├── train/           # 315 training images
    │   ├── val/             # 68 validation images
    │   └── test/            # 73 holdout test images
    │
    └── labels/              # Matching YOLO annotation text files (.txt)
        ├── train/           # 315 training label files
        ├── val/             # 68 validation label files
        └── test/            # 73 test label files
```

---

## 🛡️ 4. Split Strategy & Zero Data Leakage

To prevent optimistic bias and data leakage:
* The training, validation, and test splits are **completely disjoint** sets of images.
* Sequential and related shots from the same source sessions remain confined within their respective split.
* Verified via [`tests/test_phase3_2.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/tests/test_phase3_2.py): 0 overlapping image stems exist between `train`, `val`, and `test`.

---

## 🎯 5. The Class ID: `0 = elephant`

Our custom dataset uses a single class index:
* **`0`** corresponds to **`elephant`**
* All bounding box lines in label files begin with `0`.

The configuration file [`data.yaml`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/datasets/elephant/data.yaml) maps this:
```yaml
path: datasets/elephant
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: elephant
```

---

## 📐 6. The YOLO Annotation Format

Every image has a corresponding `.txt` label file with the **exact same stem**:

```text
Image:  datasets/elephant/images/train/elephant_train_0001.jpg
Label:  datasets/elephant/labels/train/elephant_train_0001.txt
```

### Syntax:
```text
<class_id> <x_center> <y_center> <width> <height>
```

| Field | Meaning | Valid Range |
| :--- | :--- | :--- |
| `<class_id>` | Class identifier | `0` (always `0` for elephant) |
| `<x_center>` | Normalized horizontal center | `0.0` to `1.0` |
| `<y_center>` | Normalized vertical center | `0.0` to `1.0` |
| `<width>` | Normalized box width | `0.0 < w <= 1.0` |
| `<height>` | Normalized box height | `0.0 < h <= 1.0` |

### Coordinate Normalization:
$$\text{x\_center} = \frac{X_{\min} + X_{\max}}{2 \times \text{width}_{\text{img}}}$$
$$\text{y\_center} = \frac{Y_{\min} + Y_{\max}}{2 \times \text{height}_{\text{img}}}$$
$$\text{width} = \frac{X_{\max} - X_{\min}}{\text{width}_{\text{img}}}$$
$$\text{height} = \frac{Y_{\max} - Y_{\min}}{\text{height}_{\text{img}}}$$

---

## 🌿 7. Hard Negative Background Samples

To prevent false alarms in real deployment environments, the dataset intentionally includes **74 negative images** (16.2% of the dataset):
* Images containing savanna foliage, grasslands, dirt trails, and watering holes without elephants.
* Images containing other African fauna (African buffaloes, rhinos, zebras).
* **Format**: The corresponding `.txt` file is completely empty (0 bytes).
* **Effect**: When YOLO trains on these empty-label images, it learns to suppress detections on non-target quadrupeds and natural foliage without impacting elephant recall.

---

## 🔍 8. Quality-Control Procedure

Before finalizing the dataset, the following quality checks were executed and passed:

1. **Automated Structural & Annotation Check**:
   ```powershell
   python ai/validate_dataset.py
   ```
   * Result: **`STATUS: PASSED`** (0 missing labels, 0 orphaned labels, 0 invalid annotations, 747 valid boxes).
2. **Visual Inspection**:
   Random samples across `train`, `val`, and `test` were rendered using [`ai/visualize_dataset.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/ai/visualize_dataset.py) to confirm:
   * Bounding boxes tightly enclose elephants without spatial offset.
   * Herd scenes correctly annotate individual animals separately (up to 14 elephants in a single frame).
   * Background samples correctly register as 0-box scenes.
3. **Automated Pipeline Tests**:
   Both [`tests/test_phase3_1.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/tests/test_phase3_1.py) and [`tests/test_phase3_2.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/tests/test_phase3_2.py) pass with 100% success.
4. **Phase 2 Non-Regression**:
   [`tests/test_phase2.py`](file:///c:/Users/HP/Documents/c_programm/Projects/Elephant_detector/tests/test_phase2.py) passes with zero regression.

---

## 🛠️ 9. Beginner Annotation Workflow (For Future Expansion)

When adding new field camera images to the dataset:

1. **Recommended Tool**: **Roboflow** (web-based) or **Label Studio** (self-hosted).
2. **Configuration**:
   * Create single class named `elephant` (Class ID 0).
   * Draw tight rectangular bounding boxes around every visible elephant.
   * If an elephant is partially occluded behind trees, bound the visible portion and reasonable body extent.
3. **Exporting**:
   * Export format: **YOLOv8 PyTorch TXT**.
   * Copy images to `images/{split}/` and labels to `labels/{split}/`.
4. **Validation**:
   * Run `python ai/validate_dataset.py` to confirm zero syntax or coordinate errors.

---

## ⚖️ 10. Current Dataset vs Future Improvements

### CURRENT DATASET (Phase 3.2 — Present):
* ✅ 456 total verified images (382 positive, 74 hard negatives).
* ✅ 747 ground-truth elephant bounding boxes.
* ✅ Wild African elephants in natural habitats (bush, watering holes, plains).
* ✅ High diversity of group sizes: solitary bulls, mother-calf pairs, and large family herds (up to 14 elephants).
* ✅ Hard negative coverage of other large herbivorous quadrupeds (buffalo, zebra, rhino).

### FUTURE DATASET IMPROVEMENTS (Phase 3.4+ Roadmap):
* ⏳ **Asian Elephants (*Elephas maximus*)**: Incorporate camera trap footage from South and Southeast Asian reserves (smaller ears, distinct back profile, different forestry).
* ⏳ **Night & Low-Light Imagery**: Add dawn, dusk, and infrared/night-vision footage.
* ⏳ **Peri-Urban & Agricultural Habitats**: Add images along tea/coffee plantations, railway tracks, crop fields, and village borders.
* ⏳ **Adverse Weather**: Add heavy rain and dense fog camera sequences.
