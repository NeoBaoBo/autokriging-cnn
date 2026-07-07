# AutoKriging-CNN

Automated end-to-end Kriging parameter prediction using three-dimensional convolutional neural networks.

## Overview

AutoKriging-CNN predicts nine Kriging parameters — three variogram parameters (nugget, sill, range) and six search ellipsoid parameters (major, semi-major, and minor axis lengths, plus azimuth, dip, and plunge angles) — directly from voxelized drillhole data through a single-channel 3D convolutional neural network.

**Model architecture (V3):** 4 Conv3D blocks (16→32→64→128) with BatchNorm + ReLU, followed by MaxPool3d(2) after the first three blocks and AdaptiveAvgPool3d(2,2,2) at the end. A shared FC layer (1024→256, Dropout 0.25) feeds two task-specific heads (256→64→3 for variogram, 256→64→6 for ellipsoid). All nine parameters use sigmoid-based scale-to-range output mapping.

## Files

| File | Description |
|------|-------------|
| `CNN_train.py` | Training script with multi-task regression loss and physical consistency constraints |
| `AutoKrigingNN.py` | Inference script (CLI + GUI): loads trained model and predicts Kriging parameters from drillhole Excel data |
| `random_field_generator.py` | Generates anisotropic 3D random fields via GSTools for synthetic training data |

## Installation

```bash
pip install torch numpy pandas scipy gstools tqdm tensorboard
```

## Usage

### 1. Generate synthetic training data

```bash
python random_field_generator.py
```

Outputs to `synthetic_data/`, with each sample in a subdirectory containing:
- `voxel_value.npy` — the full 3D random field
- `drillholes.csv` — sampled drillhole coordinates and values
- `parameters.json` — ground-truth variogram and ellipsoid parameters

### 2. Train the model

```bash
python CNN_train.py
```

Checkpoints and logs are saved to `trained_models/` and `training_logs/`.

### 3. Run inference

**Command line (batch):**
```bash
# Single orebody
python AutoKrigingNN.py --model best_model.pth --data drillholes.xlsx --out results/

# Batch processing
python AutoKrigingNN.py --model best_model.pth --data-dir data/ --out results/
```

Input Excel format: columns X, Y, Z, Element1, Element2, ... (first three columns = spatial coordinates, remaining columns = element grades).

**GUI:**
```bash
python AutoKrigingNN.py
```

### Important notes

- The model uses **strict=True** loading — ensure the checkpoint architecture matches exactly (1 input channel, conv/fc_v/fc_e naming).
- **V3 training data limitations**: The synthetic training data has parameter bounds: sill ∈ [0.001, 1.8], range ∈ [5, 150] m. Predictions outside these ranges will be clipped.
- The model is trained on single-structure variogram models (spherical, exponential, Gaussian). Nested multi-structure models are not represented in the current training data.

## Trained model

The trained model weights (`best_model_20260331-2.pth`) are available at:
<https://github.com/NeoBaoBo/autokriging-cnn/releases>

## Reference

Wang, Y.P. and Wang, Y.S. Automated End-to-End Prediction of Kriging Parameters Using Three-Dimensional Convolutional Neural Networks. *Computers & Geosciences*, under review.

## License

MIT License. See [LICENSE](LICENSE) for details.
