=# Brain Tumor Segmentation using U-Net++

This project implements brain tumor segmentation on the BRISC dataset using U-Net++ architecture with EfficientNet-B7 and ResNet-101 encoders combined with scSE attention mechanism.

## Results

### Overall Performance

| Model | Dice (%) | IoU (%) |
|-------|----------|---------|
| EfficientNet-B7 | 90.92 | 83.36 |
| ResNet-101 | 89.89 | 81.97 |

### Per-Tumor Type Performance

| Model | Tumor Type | Dice (%) | IoU (%) |
|-------|------------|----------|---------|
| **EfficientNet-B7** | Glioma | 89.78 | 81.46 |
| | Meningioma | 92.96 | 86.85 |
| | Pituitary | 89.96 | 81.12 |
| **ResNet-101** | Glioma | 88.34 | 79.56 |
| | Meningioma | 92.14 | 85.63 |
| | Pituitary | 88.92 | 80.27 |

## Key Findings

- **EfficientNet-B7 outperforms ResNet-101** by approximately 1 percentage point in Dice score
- **Meningioma** is the easiest to segment (92.96% Dice) due to well-defined boundaries
- **Glioma** is the most challenging (89.78% Dice) due to infiltrative nature
- External Contour Cropping (ECC) improves Dice by 3.29 percentage points
- scSE attention adds 1.65 percentage points with minimal parameter increase

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU training)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-segmentation.git
cd brain-tumor-segmentation
pip install -r requirements.txt
