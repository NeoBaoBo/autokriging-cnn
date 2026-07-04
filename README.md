# AutoKriging-CNN

End-to-end automated Kriging parameter prediction using 3D convolutional neural networks.

## Overview

AutoKriging-CNN predicts nine Kriging modeling parameters---three variogram parameters (nugget, sill, range) and six search ellipsoid parameters (three axis lengths plus azimuth, dip, and plunge angles)---directly from voxelized drillhole data.

## Files

| File | Description |
|------|-------------|
| `CNN_train.py` | Training script: 3D CNN with multi-task loss and physical consistency constraints |
| `AutoKrigingNN.py` | Inference GUI: loads trained model and predicts Kriging parameters from drillhole Excel files |
| `random_field_generator.py` | Generates anisotropic 3D random fields via GSTools for synthetic training data |

## Installation

```bash
pip install torch numpy pandas gstools tqdm tensorboard
```

## Usage

### 1. Generate training data

```bash
python random_field_generator.py
```

Outputs to `synthetic_data/`, organized as subdirectories each containing:
- `synthetic_field.npy` — the full 3D random field
- `drillholes.csv` — sampled drillhole coordinates and values
- `parameters.json` — ground-truth variogram parameters

### 2. Train the model

```bash
python CNN_train.py
```

Checkpoints and logs are saved to `trained_models/` and `training_logs/`.

### 3. Run inference

```bash
python AutoKrigingNN.py
```

Launches a GUI for loading trained model weights and drillhole Excel files.

## License

MIT License. See [LICENSE](LICENSE) for details.
