# API Reference

## Core Modules

### `AutoKrigingNN.py` — Inference Engine

#### `class AutoKrigingCNN_V3(in_channels=1)`

The V3 model architecture. Single-channel 3D CNN with 4 Conv3D blocks.

**Parameters:**
- `in_channels` (int, default=1): Number of input channels.

**Methods:**
- `forward(x)`: Forward pass. Input shape `(B, 1, D, H, W)`. Returns `(v_out, e_out)` where `v_out` has shape `(B, 3)` and `e_out` has shape `(B, 6)`.
- `scale_to_range(x, min_val, max_val)` (static): Maps raw logits to bounded output via sigmoid.

#### `load_model(model_path, device='cpu')`

Loads a trained checkpoint with `strict=True` verification.

**Parameters:**
- `model_path` (str): Path to `.pth` file.
- `device` (str): `'cpu'` or `'cuda'`.

**Returns:** AutoKrigingCNN_V3 instance in eval mode.

**Raises:** AssertionError if model architecture mismatch detected.

#### `predict(model, data_path, device='cpu')`

Runs inference on a single drillhole Excel file.

**Parameters:**
- `model`: Loaded AutoKrigingCNN_V3 instance.
- `data_path` (str): Path to drillhole `.xlsx` file.
- `device` (str): Compute device.

**Returns:** Dict with element names as keys, each containing 9 predicted parameters.

#### `voxelize_idw(x, y, z, val, target_shape=(32,32,32), k=8, power=2.0)`

IDW interpolation of drillhole data to regular 3D grid.

**Parameters:**
- `x, y, z` (ndarray): Drillhole coordinates.
- `val` (ndarray): Grade values.
- `target_shape` (tuple): Output grid dimensions.
- `k` (int): Number of nearest neighbors for IDW.
- `power` (float): Distance weighting exponent.

**Returns:** ndarray of shape `target_shape`.

#### `normalize_field(field)`

Z-score normalization of non-zero voxels.

### `CNN_train.py` — Training Pipeline

#### `class GeologicalDatasetV3(Dataset)`

PyTorch Dataset for synthetic training data.

**Parameters:**
- `data_dir` (str): Path to synthetic data directory.
- `transform` (callable, optional): Data augmentation.
- `use_mask` (bool): If True, use 2-channel input (voxel + mask).

#### `class AutoKrigingCNN_V3(nn.Module)`

Same as in AutoKrigingNN.py (training version).

**Additional method:**
- `scale_to_range(x, min_val, max_val)` (static): Sigmoid mapping.

### `random_field_generator.py` — Data Generation

#### `generate_sample(output_dir, seed=None)`

Generates a single synthetic training sample.

**Parameters:**
- `output_dir` (str): Output directory.
- `seed` (int, optional): Random seed for reproducibility.

**Outputs:**
- `voxel_value.npy`: 64×64×64 random field.
- `drillholes.csv`: Sampled drillhole data.
- `parameters.json`: Ground-truth variogram parameters.

## Parameter Ranges (V3 Training)

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| nugget | 0.0001 | 0.3 | Nugget effect |
| sill | 0.001 | 1.8 | Total sill |
| range | 5.0 | 150.0 | Variogram range (m) |
| major | 10.0 | 500.0 | Major axis (m) |
| minor | 5.0 | 300.0 | Semi-major axis (m) |
| vertical | 2.0 | 150.0 | Minor axis (m) |
| azimuth | 0.0 | 360.0 | Azimuth (degrees) |
| dip | -90.0 | 90.0 | Dip (degrees) |
| plunge | -90.0 | 90.0 | Plunge (degrees) |

## CLI Interface

```bash
# Single file inference
python AutoKrigingNN.py --model best_model.pth --data Cu-110.xlsx --out results/

# Batch inference
python AutoKrigingNN.py --model best_model.pth --data-dir data/ --out results/

# Launch GUI
python AutoKrigingNN.py
```

## Input Format

Excel file with columns:
- Column 1: X coordinate (m)
- Column 2: Y coordinate (m)
- Column 3: Z coordinate (m)
- Columns 4+: Element grades (Cu, Mo, ...)

## Output Format

Each element produces a `.xlsx` file and `summary.xlsx` containing:

| Column | Description |
|--------|-------------|
| dataset | Source filename |
| element | Element name |
| nugget | Predicted nugget |
| sill | Predicted sill |
| range | Predicted range (m) |
| major | Major axis (m) |
| semi_major | Semi-major axis (m) |
| minor | Minor axis (m) |
| azimuth | Azimuth (degrees) |
| dip | Dip (degrees) |
| plunge | Plunge (degrees) |
