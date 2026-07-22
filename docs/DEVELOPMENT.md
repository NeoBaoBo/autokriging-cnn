# Development Guide

## Environment Setup

```bash
# Create conda environment
conda create -n autokriging python=3.9
conda activate autokriging

# Install dependencies
pip install torch numpy pandas scipy gstools tqdm tensorboard openpyxl
```

## Project Structure

```
autokriging-cnn/
├── AutoKrigingNN.py           # Inference script (CLI + GUI)
├── CNN_train.py               # Training script
├── random_field_generator.py  # Synthetic data generator
├── requirements.txt           # Python dependencies
├── README.md                  # Quick start guide
├── LICENSE                    # MIT License
└── docs/
    ├── ARCHITECTURE.md        # Architecture and technical design
    ├── API_REFERENCE.md       # API documentation
    ├── DEVELOPMENT.md         # This file
    ├── FAQ.md                 # Frequently asked questions
    └── CHANGELOG.md           # Version history
```

## Running Tests

The repository includes model verification scripts:

```bash
# Verify model architecture matches checkpoint
python -c "from AutoKrigingNN import load_model; m = load_model('best_model.pth')"

# Verify inference pipeline
python -c "
from AutoKrigingNN import predict, load_model
m = load_model('best_model.pth')
r = predict(m, 'test_data.xlsx')
print(r)
"
```

## Code Style

- Python 3.9+ compatible
- Follow PEP 8 conventions
- Use type hints for public functions
- Docstrings for all public APIs
- Line length: 100 characters maximum

## Training a New Model

1. Generate synthetic data:
   ```bash
   python random_field_generator.py
   ```
   Outputs: `synthetic_data/` with 3,000 samples.

2. Train the model:
   ```bash
   python CNN_train.py
   ```
   Outputs: `trained_models/best_model.pth`, `training_logs/`.

3. Verify the model:
   ```bash
   python -c "from AutoKrigingNN import load_model; load_model('trained_models/best_model.pth')"
   ```

## Data Format

### Synthetic Training Data

Each sample in `synthetic_data/` is a subdirectory containing:
- `voxel_value.npy`: float32 array (64, 64, 64) — the complete 3D random field.
- `voxel_mask.npy` (optional): bool/int array — binary mask of sampled locations.
- `drillholes.csv`: sampled points (X, Y, Z, Grade).
- `parameters.json`: ground-truth parameters.

### Real Drillhole Data

Excel files with columns X, Y, Z, Element1, Element2, ...

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes and verify with existing trained model.
4. Submit a pull request with a clear description of changes.

## Contact

- Corresponding author: Yunsen Wang (wangyunsen@mail.neu.edu.cn)
- Issues: https://github.com/NeoBaoBo/autokriging-cnn/issues
