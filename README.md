# r2splat

面向 CT 重建的快速一阶 3D splatting 项目。当前版本只保留 Adam 一阶优化路径，并支持两类 primitive：

- `gaussian`：默认高斯核。
- `beta`：compact-support 3D Beta density，形状激活为 `gamma = 3.5 * exp(raw_beta)`；投影路径使用 CT 线积分后的指数 `gamma + 0.5`，体素化路径直接查询 3D density。

当前默认配置使用 FDK 粗重建初始化、纯 L1 图像损失，以及指数学习率衰减到初始值的 0.1 倍。

## 安装

本项目不再在运行时检测 CUDA 源码并自动 JIT 重编译。只要修改了 `gsplat2ct/cuda` 下的 C++/CUDA 源码，都需要重新安装。

开发安装：

```bash
pip install -e . --no-build-isolation
```

非开发安装：

```bash
pip install . --no-build-isolation
```

## 测试

完整数值测试：

```bash
pytest -q tests
```

只测 beta 和 TV：

```bash
pytest -q tests/test_beta_formulas.py tests/test_beta_low_level_numeric.py tests/test_tv_loss.py
```

## 训练

Gaussian primitive：

```bash
python -m scripts.train --config-name gaussian \
  exp.data_dir=/path/to/ct/scene \
  exp.result_dir=output/gaussian_quick
```

Beta primitive：

```bash
python -m scripts.train --config-name beta \
  exp.data_dir=/path/to/ct/scene \
  exp.result_dir=output/beta_quick
```

选择 primitive 数量控制策略：

```bash
# 默认 grad2d_based_adc
python -m scripts.train --config-name gaussian strategy=grad2d_based_adc ...

# Kernel-Agnostic MCMC
python -m scripts.train --config-name gaussian strategy=ka_mcmc ...
```

## 验证

```bash
python -m scripts.validate \
  data_dir=/path/to/ct/scene \
  checkpoint_path=output/gaussian_quick/ckpts/ckpt_final.pt \
  result_dir=output/gaussian_quick/validation
```

checkpoint 会保存 primitive 类型和 beta 激活参数；验证脚本会自动按 checkpoint 中的设置渲染。

## TV loss

TV loss 对 x/y/z 三个方向的 forward-difference edges 的绝对值求平均。因此默认 `lambda_tv=0.05` 不再使用 `0.05 / 3` 的临时权重补偿，也不再使用平滑项。
