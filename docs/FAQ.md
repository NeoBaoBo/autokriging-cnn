# Frequently Asked Questions

## General

**Q: What does AutoKriging-CNN predict?**
A: Nine Kriging parameters: nugget, sill, range (variogram); major axis, semi-major axis, minor axis, azimuth, dip, plunge (search ellipsoid).

**Q: What input format does it accept?**
A: Excel files with columns X, Y, Z, Element1, Element2, ... Drillhole coordinates in meters, grades as numeric values.

**Q: What output does it produce?**
A: An Excel file per element with all 9 predicted parameters, plus a `summary.xlsx` for batch runs.

## Model Architecture

**Q: Why 1 input channel instead of 2?**
A: The V3 model uses single-channel input (grade values only). Experiments showed the mask channel did not significantly improve performance.

**Q: Why Sigmoid output for all parameters?**
A: Sigmoid-based scale-to-range mapping ensures all outputs fall within geostatistically meaningful bounds: nugget ∈ [0.0001, 0.3], sill ∈ [0.001, 1.8], range ∈ [5, 150], axes ∈ domain-specific ranges, angles ∈ [0, 360] or [-90, 90].

**Q: Why are the three axis predictions sorted?**
A: Sorting ensures major ≥ semi-major ≥ minor without the aggressive chaining of min() operations used in earlier versions, which collapsed all three axes to identical values.

## Training

**Q: How many training samples were used?**
A: 3,000 synthetic 3D random fields, split 8:1:1 (train/val/test).

**Q: What variogram models were used in training?**
A: Spherical, exponential, and Gaussian single-structure models.

**Q: What are the training parameter ranges?**
A: Sill ∈ [0.5, 3.0], range ∈ [50, 300] m, anisotropy ratios ∈ [1.5, 5.0].

**Q: Can I train on my own data?**
A: Yes. Modify `random_field_generator.py` to match your deposit's parameter ranges, or create custom training samples following the same directory structure.

## Inference

**Q: How do I use the trained model?**
A: `python AutoKrigingNN.py --model best_model.pth --data drillholes.xlsx --out results/`

**Q: What if my drillhole data doesn't fill a 32x32x32 grid well?**
A: The IDW voxelization with k=8 neighbors handles sparse data. Adjust `cell_size` (currently 10m) in the voxelization step if your deposit extent differs significantly.

**Q: How long does inference take?**
A: Typically 1-2 seconds per orebody on CPU, including voxelization and model forward pass.

## Limitations

**Q: Why do sill predictions cluster near 1.8?**
A: The V3 training data has sill ∈ [0.001, 1.8]. Real deposit sills outside this range will be clipped. This is a known limitation of the V3 training distribution.

**Q: Why does the model output single-structure spherical models?**
A: The training data only includes single-structure models. Nested structures are not currently supported.

**Q: Has the model been validated on deposits other than Wushan?**
A: No. The current validation is limited to the Wushan porphyry Cu-Mo deposit. Generalization to other deposit types requires further investigation.

## Troubleshooting

**Q: Model loading fails with "AssertionError: Model mismatch"?**
A: This means the checkpoint architecture doesn't match the current code. Use `strict=True` loading and verify the checkpoint was saved from the same model version.

**Q: "Permission denied" when saving results?**
A: Close the output Excel file if it's open in another program, or change the output directory.

**Q: Memory error during voxelization?**
A: For large deposits (>50M cells), increase `cell_size` to reduce grid dimensions.
