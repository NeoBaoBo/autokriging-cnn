# Architecture and Technical Design

## Overview

AutoKriging-CNN is a 3D convolutional neural network that predicts nine Kriging parameters (three variogram + six search ellipsoid) directly from voxelized drillhole data.

## System Architecture

```
Input (32x32x32 voxel grid, 1 channel)
  │
  ├─ Conv3D Block 1: Conv3D(1→16, 3x3x3) → BN → ReLU → MaxPool(2) → (16,16,16)
  ├─ Conv3D Block 2: Conv3D(16→32, 3x3x3) → BN → ReLU → MaxPool(2) → (8,8,8)
  ├─ Conv3D Block 3: Conv3D(32→64, 3x3x3) → BN → ReLU → MaxPool(2) → (4,4,4)
  ├─ Conv3D Block 4: Conv3D(64→128, 3x3x3) → BN → ReLU          → (4,4,4)
  │   → AdaptiveAvgPool3d(2,2,2) → 128×2×2×2 = 1024
  │
  ├─ FC: Linear(1024→256) → ReLU → Dropout(0.25)
  │
  ├─ Variogram Head:   Linear(256→64) → ReLU → Linear(64→3) → Sigmoid
  │   Output: [nugget, sill, range]
  │
  └─ Ellipsoid Head:   Linear(256→64) → ReLU → Linear(64→6) → Sigmoid
      Output: [major, semi-major, minor, azimuth, dip, plunge]
```

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input dimensionality | 3D (32x32x32) | Deposit modeling is inherently 3D; 2D approaches lose vertical anisotropy information |
| Input channels | 1 (grade values only) | V3 model uses single-channel input; the mask channel did not improve performance |
| Architecture | 4-layer Conv3D | Progressive channel expansion (16→32→64→128) captures multi-scale spatial features |
| Pooling | MaxPool(2) ×3, AdaptiveAvgPool(2,2,2) | Fixed output dimension regardless of input size |
| Output activation | Sigmoid (scale-to-range) | All 9 outputs bounded to geostatistically meaningful intervals |
| Axis ordering | Sort descending after sigmoid | Ensures major ≥ semi-major ≥ minor without aggressive min() chaining |
| Loss function | Composite (Smooth L1 + physical penalties) | Balances regression accuracy with geostatistical validity |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) | Standard choice for CNN regression |
| Learning rate schedule | ReduceLROnPlateau (patience=8, factor=0.5) | Stable convergence without manual tuning |
| Voxelization | Inverse Distance Weighting (IDW, k=8) | Smooth interpolation of sparse drillhole data |
| Framework | PyTorch | Industry-standard deep learning framework with strong 3D convolution support |

## Data Flow

```
Drillhole Excel (X, Y, Z, Grade)
  → Data cleaning (remove NaN, string values)
  → IDW voxelization (cKDTree, k=8 nearest neighbors)
  → Z-score normalization (non-zero voxels)
  → PyTorch tensor [1, 1, 32, 32, 32]
  → CNN forward pass
  → Sigmoid scale-to-range mapping
  → Axis sorting (descending)
  → Output: 9 Kriging parameters
```

## Training Data Pipeline

```
GSTools random field generator
  → 64×64×64 anisotropic 3D fields (spherical/exp/Gaussian)
  → Simulated drillhole sampling
  → 32×32×32 IDW voxelization
  → parameters.json (ground truth labels)
  → GeologicalDatasetV3 (PyTorch Dataset)
  → DataLoader (batch_size=16, 8:1:1 split)
```

## Dependencies

- PyTorch ≥ 2.0.0: Deep learning framework
- NumPy ≥ 1.24.0: Numerical computation
- Pandas ≥ 2.0.0: Data loading and processing
- SciPy ≥ 1.10.0: Spatial algorithms (cKDTree)
- GSTools ≥ 1.5.0: Geostatistical random field generation
- Matplotlib: Visualization
- TensorBoard: Training monitoring
- openpyxl: Excel file I/O
