# AutoKriging-CNN

Automated prediction of the variogram range for Kriging using a three-dimensional convolutional neural network.

## Overview

AutoKriging-CNN predicts the **variogram range along 64 directions** directly from voxelized drillhole data. The three axis lengths and principal orientations of the search ellipsoid are then recovered from these directional ranges by positive-definite tensor fitting, and the sill and nugget are computed analytically (the sill as the sample variance and the nugget as the experimental variogram intercept). This design lets the network concentrate on the second-order spatial structure (the range), which is the quantity that actually requires variogram fitting.

## Files

| File | Description |
|------|-------------|
| `AutoKrigingNN.py` | The 3D CNN model, the directional-range (Fibonacci sphere) sampler, and the positive-definite tensor fitting that recovers the axis lengths and orientations |
| `CNN_train.py` | Training script: builds the z-score-normalized dataset and trains the model with a mean-squared-error loss over the 64 directional ranges |
| `random_field_generator.py` | Generates anisotropic 3D random fields via GSTools (scale-down field geometry, simulated drillholes, IDW voxelization, 64-direction range labels) for synthetic training data |

## Installation

```bash
pip install torch numpy pandas scipy gstools
```

## Usage

### 1. Generate synthetic training data

```bash
python random_field_generator.py --out_dir synthetic_data --n_samples 3000
```

The synthetic fields follow the deposit geometry scaled down by a factor of three (450 × 450 × 282 m), with a major range of 50–100 m, near-isotropic axis ratios (minor/major 0.6–1.0, vertical/major 0.4–1.0), and a full dip range of ±90°. Each sample is saved in a subdirectory containing:

- `idw_grid.npy` — the IDW-reconstructed 32×32×32 grade grid
- `mask_grid.npy` — the validity mask
- `ranges.npy` — the 64 directional range labels
- `drillholes.csv` — the simulated drillhole samples
- `parameters.json` — the ground-truth parameters

### 2. Train the model

```bash
python CNN_train.py --data_dir synthetic_data --save_dir trained_models --epochs 300
```

The model is trained for 300 epochs with the AdamW optimizer and a cosine annealing schedule. The best checkpoint (`best_model.pth`) and the training history are saved to the output directory.

### 3. Run inference

```python
import torch
from AutoKrigingNN import AutoKrigingNN, N_DIR, DEVICE, fibonacci_sphere, fit_tensor_psd, tensor_to_axes_angles

model = AutoKrigingNN(out_dim=N_DIR).to(DEVICE)
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.eval()

# x: [1, 2, 32, 32, 32] tensor (idw_grid + mask_grid, z-score normalized)
ranges = model(x)                       # 64 directional ranges
axes, eigvecs = tensor_to_axes_angles(fit_tensor_psd(fibonacci_sphere(N_DIR), ranges))
# axes: [major, semi-major, minor]; eigvecs: principal orientations
```

## Important notes

- **Scale-down factor.** The training fields are scaled down by a factor of three relative to the deposit. At inference on real data, multiply the predicted range (and the recovered axis lengths) by three to recover field-scale values.
- **Single-structure model.** The recovered variogram is a single-structure spherical model whose range approximates the expert long-range component. Nested (multi-structure) variograms are not represented in the current training data and are noted as a limitation.
- **Analytic sill and nugget.** The sill is the sample variance and the nugget is the experimental variogram intercept; they are computed outside the network and are element-dependent.

## Reference

Wang, Y.P., Zhang, H.T., and Wang, Y.S. Automated End-to-End Prediction of Kriging Parameters Using Three-Dimensional Convolutional Neural Networks. *Applied Computing and Geosciences*, under review.

## License

MIT License. See [LICENSE](LICENSE) for details.
