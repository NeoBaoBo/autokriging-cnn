# Architecture and Technical Design

## Overview

AutoKriging-CNN predicts the **variogram range along 64 directions** directly from voxelized drillhole data using a four-layer 3D convolutional neural network. The three axis lengths and principal orientations of the search ellipsoid are then recovered from these directional ranges by positive-definite tensor fitting. The sill and nugget are computed analytically outside the network — the sill as the sample variance and the nugget as the experimental variogram intercept — so the network concentrates on the second-order spatial structure (the range), which is the quantity that actually requires variogram fitting.

## System Architecture

```
Input (32×32×32, 2 channels: IDW-reconstructed grade + validity mask)
  │
  ├─ Conv3D Block 1: Conv3D(2→16, 3×3×3) → BN → ReLU → MaxPool(2)   → (16,16,16)
  ├─ Conv3D Block 2: Conv3D(16→32, 3×3×3) → BN → ReLU → MaxPool(2)  → (8,8,8)
  ├─ Conv3D Block 3: Conv3D(32→64, 3×3×3) → BN → ReLU → MaxPool(2)  → (4,4,4)
  ├─ Conv3D Block 4: Conv3D(64→128, 3×3×3) → BN → ReLU → AdaptiveAvgPool(1,1,1)
  │
  └─ Regression head: Flatten → Linear(128→128) → ReLU → Linear(128→64)
      Output: 64 directional ranges
```

The network outputs a single vector of 64 directional ranges. The recovery of the search ellipsoid happens outside the network:

```
64 directional ranges  {r_i} along Fibonacci-sphere directions {u_i}
  → fit_tensor_psd:  solve u_iᵀ D u_i = 1 / r_i²  with D = L Lᵀ (positive definite)
  → tensor_to_axes_angles: eigendecomposition of D
  → three axis lengths (major, semi-major, minor) + principal orientations
```

## Directional Range Representation

The directional range of an anisotropic ellipsoid with axis lengths `a₁ ≥ a₂ ≥ a₃` satisfies

```
r(u) = 1 / sqrt(uᵀ D u)
```

where `D` is the symmetric positive-definite anisotropy tensor. Given predicted directional ranges, `D` is recovered by least-squares fitting of `1/r_i² = u_iᵀ D u_i` under a positive-definite constraint. `D` is parameterized through its Cholesky factor `D = L Lᵀ`, which guarantees positive eigenvalues and hence finite, physically meaningful axis lengths even when the predictions are noisy. Eigendecomposition then yields the axis lengths `a_k = 1/√λ_k` and the principal orientations.

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input dimensionality | 3D (32×32×32) | Deposit modeling is inherently 3D; 2D approaches lose vertical continuity information |
| Input channels | 2 (IDW grade + validity mask) | The mask distinguishes sampled vs. void voxels, stabilizing learning on sparse drillhole data |
| Architecture | 4-layer Conv3D | Progressive channel expansion (2→16→32→64→128) captures multi-scale spatial features |
| Pooling | MaxPool(2) ×3, AdaptiveAvgPool(1,1,1) | Fixed 128-dim output regardless of input size |
| Output head | Single linear head (128→128→64) | One regression target — the 64 directional ranges — no multi-task branching |
| Loss function | MSE over 64 directional ranges | Direct regression of the continuity scale, no physical-constraint penalties |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) | Standard choice for CNN regression |
| Learning rate schedule | Cosine annealing (→ 1e-5) | Smooth decay with stable late-stage convergence |
| Voxelization | Inverse Distance Weighting (IDW, k=12, power=2.0) | Smooth interpolation of sparse drillhole data |
| Tensor fitting | Positive-definite (Cholesky `D = L Lᵀ`) | Guarantees finite axis lengths even for noisy range predictions |
| Sill / nugget | Analytic (sample variance / variogram intercept) | Zero-order statistics are computed, not regressed |

## Data Flow

```
Drillhole samples (X, Y, Z, Grade)
  → IDW voxelization (cKDTree, k=12, power=2.0) → 32×32×32 grade grid + validity mask
  → z-score normalization of valid voxels (at training and inference time)
  → PyTorch tensor [1, 2, 32, 32, 32]
  → CNN forward pass → 64 directional ranges
  → fit_tensor_psd → tensor_to_axes_angles → axis lengths + orientations
  → (inference on real data) multiply range and axes by 3 to recover field scale
```

The sill is the sample variance of the grades, and the nugget is the intercept of the experimental variogram at zero lag; both are computed directly from the drillhole data in a single pass.

## Training Data Pipeline

```
random_field_generator.py
  → GSTools anisotropic 3D random fields, field geometry scaled down by a factor of three
     (450×450×282 m on a 90×90×94 grid, 5×5×3 m spacing)
  → Simulated drillhole sampling (32 m horizontal spacing, 1 m vertical interval)
  → 32×32×32 IDW voxelization (k=12) + validity mask
  → 64 directional range labels (Fibonacci sphere)
  → parameters.json (ground-truth parameters)
  → RangeDataset (z-score applied at load time)
  → DataLoader (batch_size=32, 8:1:1 split)
```

## Dependencies

- PyTorch ≥ 2.0.0: Deep learning framework
- NumPy ≥ 1.24.0: Numerical computation
- Pandas ≥ 2.0.0: Data loading and processing
- SciPy ≥ 1.10.0: Spatial algorithms (cKDTree) and tensor fitting (least_squares)
- GSTools ≥ 1.5.0: Geostatistical random field generation
