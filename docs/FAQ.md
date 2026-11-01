# Frequently Asked Questions

## General

**Q: What does AutoKriging-CNN predict?**
A: The variogram **range along 64 directions** on the unit sphere. The three axis lengths and principal orientations of the search ellipsoid are recovered from these ranges by positive-definite tensor fitting; the sill and nugget are computed analytically (not predicted).

**Q: What input format does it accept?**
A: Drillhole data with 3D coordinates and element grades. At inference the data are voxelized to a 32×32×32 grid via inverse distance weighting and fed to the network as a 2-channel tensor (grade + validity mask).

**Q: What output does it produce?**
A: 64 directional ranges, from which the search-ellipsoid axis lengths (major, semi-major, minor) and principal orientations are recovered. The sill is the sample variance and the nugget is the experimental variogram intercept.

## Model Architecture

**Q: Why 2 input channels instead of 1?**
A: Channel 0 is the z-score-normalized IDW grade grid, channel 1 is the validity mask. The mask tells the network which voxels contain real data and which are void, which stabilizes learning on sparse drillhole data.

**Q: Why is the output a single 64-dimensional vector?**
A: The anisotropy is modeled directly through 64 directional ranges rather than nine independent parameters. A single regression head predicts all directions, and the ellipsoid is recovered by tensor fitting afterward.

**Q: Why compute the sill and nugget analytically instead of predicting them?**
A: The sill is the sample variance and the nugget is the experimental variogram intercept — both are zero-order statistics with direct analytic definitions. Computing them removes the element-scale dependence from the network and lets it concentrate on the second-order spatial structure (the range), which is the quantity that actually requires variogram fitting.

## Training

**Q: How many training samples were used?**
A: 3,000 synthetic 3D random fields, split 8:1:1 (train/val/test).

**Q: What variogram models were used in training?**
A: Spherical, exponential, and Gaussian single-structure models.

**Q: What are the training parameter ranges?**
A: Major range ∈ [50, 100] m (scaled-down; 150–300 m at field scale), minor/major ∈ [0.6, 1.0], vertical/major ∈ [0.4, minor/major], sill ∈ [0.3, 1.0], azimuth ∈ [0°, 360°], dip ∈ [−90°, 90°], plunge ∈ [−30°, 30°]. The near-isotropic axis ratios reflect the weak anisotropy of porphyry mineralization.

**Q: What is the scale-down factor?**
A: The synthetic fields follow the deposit geometry scaled down by a factor of three. At inference on real data, multiply the predicted range and recovered axis lengths by three to recover field-scale values.

## Inference

**Q: How do I use the trained model?**
A: See the inference example in `README.md`. The workflow is: IDW voxelization → z-score normalization → CNN forward pass → tensor fitting → axis lengths and orientations.

**Q: What IDW parameters are used for voxelization?**
A: k=12 nearest neighbors and power=2.0 (inverse-squared distance weighting).

**Q: How long does inference take?**
A: Typically seconds per orebody, including voxelization and the model forward pass.

## Limitations

**Q: Does the model support nested (multi-structure) variograms?**
A: No. The recovered variogram is a single-structure spherical model whose range approximates the expert long-range component. Nested structures are noted as a limitation.

**Q: Has the model been validated on deposits other than Wushan?**
A: No. The current validation is limited to the Wushan porphyry Cu-Mo deposit. Generalization to other deposit types requires further investigation.

**Q: Is the sill accurate for every element?**
A: The sill is the sample variance of the grades, so it is element-specific and correct by construction. Applications requiring the Kriging variance (rather than point estimates) may need additional calibration of the sill and nugget.
