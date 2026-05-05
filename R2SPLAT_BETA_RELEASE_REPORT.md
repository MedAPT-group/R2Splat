# r2splat release 整理与 Beta primitive 特性报告

## 本次完成内容

1. TV loss 已与 r2_gaussian 对齐：
   - forward 使用 x/y/z 三个方向 forward-difference edge 的总数归一化。
   - backward 使用同一个 edge-count 归一化。
   - 当前实现使用非平滑 `abs(diff)`，已移除 TV 平滑参数。
   - `configs/trainer/default.yaml` 与 `configs/trainer/r2gaussian.yaml` 的 `lambda_tv` 恢复为 `0.05`。

2. 新增 Beta primitive 建模路径：
   - 新增 raw beta 参数 `beta`，激活为 `gamma = 3.5 * exp(beta)`。
   - 默认 `gamma_init=3.5`，因此 raw beta 为 0 时投影指数为 `gamma + 0.5 = 4.0`。
   - Gaussian primitive 仍是默认路径，配置 `model=beta` 后切换到 Beta primitive。

3. 新增 CUDA 底层算子：
   - `beta_rasterize_to_pixels_fwd/bwd`：投影路径使用 `opacity * mu_beta * (1 - d2)_+^beta2d`。
   - `beta_voxelize_to_voxels_fwd/bwd`：体素路径使用 `opacity * (1 - q)_+^gamma`。
   - 投影路径复用现有 `conics_mu` 的 Gaussian path factor，并乘上 `B(1/2,gamma+1)/sqrt(2*pi)` 得到 Beta CT 路径积分因子。

4. Python 兼容层和上层接口：
   - `gsplat2ct/model/beta.py` 提供 beta 激活、路径因子、digamma 梯度工具。
   - `GaussianParams` 支持可选 beta 参数，并在保存/加载 checkpoint 时保留 primitive 类型和 beta 激活配置。
   - 训练、投影评估、体素评估、独立验证脚本均支持 checkpoint 中的 Beta primitive。

5. 高斯核数量控制策略整理：
   - `strategy=none`：不做数量控制。
   - `strategy=default` 或 `type=grad2d_based_adc`：现有 grad2d-based ADC。
   - `strategy=ka_mcmc`：新增 Kernel-Agnostic MCMC 风格策略，支持 relocate 与采样增长。

6. 配置与损失接口收敛：
   - 移除残差正负核补采样策略及其 Hydra 配置入口。
   - 默认初始化改为 `init.recon_method=fdk`。
   - 默认 scheduler 改为指数衰减，最终学习率系数 `min_factor=0.1`。
   - 图像训练损失只保留 mean L1，已移除 L2/SSIM 训练损失接口。

## 关键实现文件

- TV loss：
  - `gsplat2ct/cuda/csrc/Loss.cpp`
  - `gsplat2ct/cuda/csrc/VolumeLossFwd.cu`
  - `gsplat2ct/cuda/csrc/VolumeLossBwd.cu`

- Beta primitive：
  - `gsplat2ct/model/beta.py`
  - `gsplat2ct/model/gaussian_params.py`
  - `gsplat2ct/ops/gs3d_ct_train.py`
  - `gsplat2ct/ops/gs3d_ct_rasterize.py`
  - `gsplat2ct/ops/gs3d_ct_voxelize.py`
  - `gsplat2ct/cuda/csrc/RasterizeToPixels*`
  - `gsplat2ct/cuda/csrc/VoxelizeToVoxels*`
  - `gsplat2ct/cuda/include/Ops.h`
  - `gsplat2ct/cuda/csrc/ext.cpp`

- 策略：
  - `gsplat2ct/strategy/default.py`
  - `gsplat2ct/strategy/mcmc.py`
  - `gsplat2ct/strategy/__init__.py`
  - `configs/strategy/ka_mcmc.yaml`

## 数值测试

新增测试：

- `tests/test_tv_loss.py`：CUDA TV forward/backward 对齐 PyTorch edge-mean reference。
- `tests/test_beta_formulas.py`：beta 激活、path ratio、digamma 梯度检查。
- `tests/test_beta_low_level_numeric.py`：beta rasterize/voxelize CUDA forward 对齐手写 PyTorch reference。
- `tests/test_beta_operator_smoke.py`：完整 beta 训练算子能返回 `means/quats/scale_ori/opacity/beta` 梯度。

已运行：

```bash
pip install -e . --no-build-isolation
python -m compileall gsplat2ct scripts setup.py
pytest -q tests
```

结果：

```text
8 passed
```

## 使用命令

安装：

```bash
pip install -e . --no-build-isolation
```

Gaussian 训练：

```bash
python -m scripts.train --config-name default \
  exp.data_dir=/path/to/ct/scene \
  exp.result_dir=output/gaussian_quick
```

Beta 训练：

```bash
python -m scripts.train --config-name default model=beta \
  exp.data_dir=/path/to/ct/scene \
  exp.result_dir=output/beta_quick
```

Beta + KA-MCMC：

```bash
python -m scripts.train --config-name default model=beta strategy=ka_mcmc \
  exp.data_dir=/path/to/ct/scene \
  exp.result_dir=output/beta_mcmc_quick
```

验证：

```bash
python -m scripts.validate \
  data_dir=/path/to/ct/scene \
  checkpoint_path=output/beta_quick/ckpts/ckpt_final.pt \
  result_dir=output/beta_quick/validation
```

## 注意事项

- 修改 CUDA/C++ 源码后必须重新运行 `pip install -e . --no-build-isolation`。
- Beta voxelization 没有路径长度因子，也没有 `B(1/2,gamma+1)`，只使用 3D density 查询。
- Beta projection 的 `mu_beta` 通过 `conics_mu` 的 Gaussian path factor 乘 ratio 得到，因此 covariance 路径梯度仍由现有 `conics_mu_bwd` 负责，gamma 路径梯度在 Python 层显式累加。
