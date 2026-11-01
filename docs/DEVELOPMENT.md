# Development Guide

## Environment Setup

```bash
# Create conda environment
conda create -n autokriging python=3.9
conda activate autokriging

# Install dependencies
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `pandas`, `scipy`, `gstools`.

## Project Structure

```
autokriging-cnn/
├── AutoKrigingNN.py           # 3D CNN model + tensor-fitting utilities
├── CNN_train.py               # Training script (z-score input, 64-direction ranges)
├── random_field_generator.py  # Synthetic data generator (scale-down field, IDW, range labels)
├── requirements.txt           # Python dependencies
├── README.md                  # Quick start guide
├── LICENSE                    # MIT License
└── docs/
    ├── ARCHITECTURE.md        # Architecture and technical design
    ├── API_REFERENCE.md       # API documentation
    ├── DEVELOPMENT.md         # This file
    └── FAQ.md                 # Frequently asked questions
```

## Training a New Model

1. Generate synthetic data:

   ```bash
   python random_field_generator.py --out_dir synthetic_data_v10 --n_samples 3000
   ```

   Outputs `synthetic_data_v10/` with 3,000 samples, each a `case_XXXXXX/` subdirectory.

2. Train the model:

   ```bash
   python CNN_train.py --data_dir synthetic_data_v10 --save_dir trained_models --epochs 300
   ```

   Outputs in `trained_models/`:
   - `best_model.pth` — best checkpoint by validation loss.
   - `training_history.csv` — per-epoch loss and learning rate.
   - `eval_result.json` — test-set axis-length and principal-orientation errors.

3. Verify the model (load the checkpoint and check the axis errors in `eval_result.json`):

   ```python
   import torch
   from AutoKrigingNN import AutoKrigingNN, N_DIR, DEVICE
   m = AutoKrigingNN(out_dim=N_DIR).to(DEVICE)
   m.load_state_dict(torch.load('trained_models/best_model.pth', weights_only=True))
   ```

## Code Style

- Python 3.9+ compatible
- Follow PEP 8 conventions
- Docstrings for all public APIs
- Line length: 100 characters maximum

## Data Format

### Synthetic Training Data

Each sample in the output directory is a subdirectory `case_XXXXXX/` containing:

- `idw_grid.npy`: float32 array `(32, 32, 32)` — the IDW-reconstructed grade grid.
- `mask_grid.npy`: float32 array `(32, 32, 32)` — the validity mask.
- `ranges.npy`: float32 array `(64,)` — the 64 directional range labels.
- `drillholes.csv`: simulated drillhole samples (`x, y, z, value`).
- `parameters.json`: ground-truth parameters (model type, sill, nugget, axes, angles).

### Real Drillhole Data

The inference pipeline reads drillhole coordinates and element grades, reconstructs the voxel grid via IDW, and feeds the resulting tensor to the network. See the inference example in `README.md`.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes and verify with the existing training pipeline.
4. Submit a pull request with a clear description of changes.

## Contact

- Corresponding author: Yunsen Wang (wangyunsen@mail.neu.edu.cn)
- Issues: https://github.com/NeoBaoBo/autokriging-cnn/issues
