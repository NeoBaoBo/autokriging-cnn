# Changelog

## v1.1.0 (2026-07-21) — Current

### Fixed
- Corrected model architecture in `AutoKrigingNN.py`: single-channel input (was incorrectly 2-channel in previous release).
- All output activations use Sigmoid scale-to-range mapping (was incorrectly Softplus/Tanh in previous release).
- `strict=True` loading enforced to prevent silent model mismatch errors.
- Axis sorting replaces chained `min()` operations to prevent axis collapse.

### Added
- CLI interface for batch inference (`--data`, `--data-dir`, `--model`, `--out`).
- IDW voxelization with cKDTree acceleration (`scipy.spatial`).
- Comprehensive documentation in `docs/`: architecture, API reference, development guide, FAQ.

### Changed
- Inference script rewritten with modular functions (`predict`, `load_model`, `voxelize_idw`).
- Output format includes `summary.xlsx` and `summary.json` for batch runs.

## v1.0.0 (2026-03-31) — Initial Release

### Added
- V3 model architecture: 4-layer Conv3D, 1-channel input, 9 outputs.
- Training script with composite multi-task loss and physical consistency constraints.
- `random_field_generator.py` for synthetic data generation via GSTools.
- GUI-based inference application.
- MIT License.

### Known Issues (fixed in v1.1.0)
- Inference script used 2-channel model architecture, mismatched with 1-channel checkpoint.
- `strict=False` loading silently accepted architecture mismatches.
- Output activation differed between training (Sigmoid) and inference (Softplus/Tanh) descriptions.
