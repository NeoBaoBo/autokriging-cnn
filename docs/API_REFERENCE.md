# API Reference

## Core Modules

### `AutoKrigingNN.py` — Model and Tensor-Fitting Utilities

#### Constants

- `N_DIR = 64`: Number of directional ranges predicted by the network.
- `DEVICE`: `torch.device`, `cuda` if available, otherwise `cpu`.

#### `fibonacci_sphere(n)`

Generates `n` approximately uniformly distributed directions on the unit sphere (Fibonacci-sphere sampling).

**Parameters:**
- `n` (int): Number of directions.

**Returns:** `ndarray` of shape `(n, 3)` with unit-length direction vectors.

#### `class AutoKrigingNN(nn.Module)`

Four-layer 3D convolutional network that predicts `N_DIR` directional ranges.

**Parameters:**
- `out_dim` (int, default=`N_DIR`): Number of output directions.

**Forward:**
- Input: `x` of shape `(B, 2, 32, 32, 32)` — channel 0 is the z-score-normalized IDW grade grid, channel 1 is the validity mask.
- Output: tensor of shape `(B, out_dim)` — the predicted directional ranges (meters, in the scaled-down field).

#### `fit_tensor_psd(directions, ranges)`

Fits a symmetric positive-definite anisotropy tensor `D` from directional ranges under a Cholesky parameterization `D = L Lᵀ`.

**Parameters:**
- `directions` (ndarray): Shape `(N, 3)` unit direction vectors.
- `ranges` (ndarray): Shape `(N,)` directional ranges.

**Returns:** `ndarray` of shape `(3, 3)`, the positive-definite tensor `D`.

#### `tensor_to_axes_angles(D)`

Recovers the axis lengths and principal orientations from a positive-definite tensor by eigendecomposition.

**Parameters:**
- `D` (ndarray): Shape `(3, 3)` positive-definite tensor.

**Returns:** tuple `(axes, eigvecs)` where `axes` is `[major, semi-major, minor]` and `eigvecs` is a `(3, 3)` matrix of principal directions (columns).

#### `angle_between(v1, v2)`

Angle in degrees between two direction vectors (treats axes as undirected by taking the absolute dot product).

**Parameters:**
- `v1`, `v2` (ndarray): Direction vectors.

**Returns:** `float` angle in degrees.

### `CNN_train.py` — Training Pipeline

#### `class RangeDataset(Dataset)`

PyTorch dataset over synthetic training samples. Applies z-score normalization to the valid voxels of the IDW grid at load time (consistent with inference).

**Parameters:**
- `data_dir` (str): Directory containing `case_*` subdirectories.

Each sample yields `(x, y, angles)` where `x` is a `(2, 32, 32, 32)` tensor, `y` is a `(64,)` tensor of directional ranges, and `angles` is `(azimuth, dip, plunge)`.

#### `evaluate_loss(model, loader)`

Computes mean MSE loss over a data loader in eval mode.

#### CLI

```bash
python CNN_train.py --data_dir synthetic_data_v10 --save_dir trained_models --epochs 300 --batch_size 32
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `synthetic_data_v10` | Training data directory |
| `--save_dir` | `trained_models` | Output directory |
| `--epochs` | `300` | Maximum training epochs |
| `--batch_size` | `32` | Batch size |

**Outputs** (in `--save_dir`):
- `best_model.pth` — best-checkpoint model weights (by validation loss).
- `training_history.csv` — per-epoch training/validation loss and learning rate.
- `eval_result.json` — test-set axis-length and principal-orientation errors.

### `random_field_generator.py` — Synthetic Data Generation

#### Global configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `INPUT_SIZE` | `(32, 32, 32)` | Voxel grid size fed to the CNN |
| `RAW_GRID_SPACING` | `(5.0, 5.0, 3.0)` | Cell spacing of the raw random field (m) |
| `BIG_GRID_SHAPE` | `(90, 90, 94)` | Raw field grid, i.e. 450×450×282 m |
| `HOLE_SPACING` | `32.0` | Horizontal drillhole spacing (m) |
| `MAJOR_MIN` / `MAJOR_MAX` | `50.0` / `100.0` | Major range bounds (m, scaled-down) |
| `MINOR_RATIO_MIN` / `MAX` | `0.6` / `1.0` | minor/major ratio bounds |
| `VERTICAL_RATIO_MIN` | `0.4` | Lower bound of vertical/major |
| `SILL_MIN` / `SILL_MAX` | `0.3` / `1.0` | Sill bounds |
| `NUGGET_RATIO` | `0.08` | Nugget = sill × ratio |

#### `generate_sgs_field(params, grid_shape, spacing, seed=None)`

Generates an anisotropic 3D random field with GSTools.

**Parameters:**
- `params` (dict): Keys `model_type`, `sill`, `nugget`, `major`, `minor`, `vertical`, `azimuth`, `dip`, `plunge`.
- `grid_shape` (tuple): Raw field shape.
- `spacing` (tuple): Cell spacing (m).
- `seed` (int, optional): Random seed.

**Returns:** `ndarray` of shape `grid_shape`.

#### `simulate_drillholes(field, spacing, hole_spacing_xy, sample_interval_z, missing_rate, depth_ratio, seed)`

Simulates regular drillhole sampling of a random field.

**Returns:** tuple `(drill_df, meta)` where `drill_df` is a DataFrame with columns `x, y, z, value`.

#### `idw_interpolation_3d(drill_df, target_grid_shape, k=12, power=2.0)`

Inverse-distance-weighted interpolation of drillhole data to a regular 3D grid.

**Parameters:**
- `drill_df` (DataFrame): Columns `x, y, z, value`.
- `target_grid_shape` (tuple): Output grid shape, default `(32, 32, 32)`.
- `k` (int, default=12): Number of nearest neighbors.
- `power` (float, default=2.0): Distance weighting exponent.

**Returns:** tuple `(value_grid, mask_grid)` — the IDW grade grid and a validity mask (voxels whose nearest neighbor lies within the 70th percentile of nearest-neighbor distances).

#### `directional_ranges(azimuth, dip, plunge, major, minor, vertical, directions)`

Computes the range along each direction for an anisotropic ellipsoid.

**Returns:** `ndarray` of shape `(len(directions),)`.

#### `gen_case(args)` / `gen_data(group_dir, n, base_seed, n_workers)`

Generate a single sample / a full dataset in parallel.

#### CLI

```bash
python random_field_generator.py --out_dir synthetic_data_v10 --n_samples 3000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--out_dir` | `synthetic_data_v10` | Output directory |
| `--n_samples` | `3000` | Number of samples |
| `--n_workers` | CPU cores − 1 | Parallel processes |

## Data Format

### Synthetic training sample

Each sample is a subdirectory `case_XXXXXX/` containing:

| File | Description |
|------|-------------|
| `idw_grid.npy` | `(32, 32, 32)` float32 IDW grade grid |
| `mask_grid.npy` | `(32, 32, 32)` float32 validity mask |
| `ranges.npy` | `(64,)` float32 directional range labels |
| `drillholes.csv` | Simulated drillhole samples (x, y, z, value) |
| `parameters.json` | Ground-truth parameters (model type, sill, nugget, axes, angles) |

### Real drillhole data

The inference pipeline reads drillhole coordinates and element grades, reconstructs the voxel grid via IDW, and feeds the resulting tensor to the network. See the inference example in `README.md`.
