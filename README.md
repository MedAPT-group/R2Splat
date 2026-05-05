# r2splat

Fast first-order 3D Gaussian Splatting for CT reconstruction.

This release focuses on the Adam training path used for CT reconstruction. The
old experimental optimizer/demo paths and runtime CUDA rebuild path have been
removed.

## Install

```bash
pip install -e . --no-build-isolation
```

After changing CUDA or C++ sources, reinstall with the same command so the
extension is rebuilt.

## Train

```bash
python -m scripts.train --config-name r2gaussian_train \
  exp.data_dir=/path/to/ct/scene \
  exp.result_dir=output/quick_check
```

The default R2Gaussian-style trainer uses `lambda_tv=0.0166666667`, matching
R2Gaussian's `0.05` TV strength after accounting for this project's
voxel-count-normalized TV implementation.

## Validate

```bash
python -m scripts.validate \
  data_dir=/path/to/ct/scene \
  checkpoint_path=output/quick_check/ckpt_29999.pt \
  result_dir=output/quick_check/validation
```
