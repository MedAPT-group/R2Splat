# CUDA MLP Classification Implementation

## 概述

这是一个完整的 MLP 分类网络的 CUDA 实现，包含前向传播、一阶梯度和二阶 Hessian 计算。

### 网络架构

```
输入 x (i×1)
  ↓
第一层: W1 (m×i) → a1 = W1*x + b1
  ↓
激活: z1 = tanh(a1)
  ↓
第二层: W2 (o×m) → a2 = W2*z1 + b2 (logits)
  ↓
Softmax: σ = exp(a2) / sum(exp(a2))
  ↓
交叉熵损失: L = -sum(y * log(σ))
```

## 文件结构

```
gsplat2ct/cuda/
├── csrc/
│   ├── mlp_classification_fwd.cu   # 前向传播实现
│   └── mlp_classification_bwd.cu   # 反向传播实现
└── include/
    └── mlp_classification.cuh      # 函数声明

tests/
├── test_demo_mlp.py                # PyTorch 参考实现
└── test_cuda_mlp.py                # CUDA 测试代码

compile_mlp_cuda.sh                 # 编译脚本
```

## 核心算法

### 前向传播

1. **第一层线性变换**: `a1 = W1 * x + b1`
2. **Tanh 激活**: `z1 = tanh(a1)`
3. **第二层线性变换**: `a2 = W2 * z1 + b2`
4. **Softmax（数值稳定版本）**:
   ```
   z_max = max(a2)
   σ = exp(a2 - z_max) / sum(exp(a2 - z_max))
   ```
5. **交叉熵损失**: `L = -sum(y * log(σ + ε))`

### 一阶梯度（反向传播）

使用优雅的梯度公式：

1. **输出层误差**: `error = σ - y` （softmax + 交叉熵的组合导数）
2. **第二层梯度**:
   - `grad_w2 = error @ z1^T`
   - `grad_b2 = error`
3. **反向传播到第一层**:
   - `grad_z1 = W2^T @ error`
   - `grad_a1 = grad_z1 * (1 - z1^2)` （tanh 导数）
4. **第一层梯度**:
   - `grad_w1 = grad_a1 @ x^T`
   - `grad_b1 = grad_a1`

### 二阶 Hessian

1. **H_b2（完整矩阵，o×o）**:
   ```
   H_b2 = diag(σ) - σσ^T
   ```

2. **H_b1（完整矩阵，m×m）**:
   ```
   H_b1 = D * (W2^T * H_b2 * W2) * D + second_term
   其中：
   - D = diag(1 - z1^2)
   - second_term = -2 * diag(s * z1 * d)
   - s = W2^T @ (σ - y)
   ```

3. **H_w2（对角近似，o*m）**:
   ```
   hess_w2_diag[i*m + j] = hess_b2[i, i] * z1[j]^2
   ```

4. **H_w1（对角近似，m*i）**:
   ```
   hess_w1_diag[i*inp + j] = hess_b1[i, i] * x[j]^2
   ```

## 编译和使用

### 1. 编译 CUDA 代码

```bash
cd /home/chengzyi/myterminus/code/gsplat-3dgs2ct
bash compile_mlp_cuda.sh
```

**注意**: 根据你的 GPU 架构修改 `compile_mlp_cuda.sh` 中的 `ARCH_FLAGS`:
- V100: `-arch=sm_70`
- A100: `-arch=sm_80`
- RTX 3090: `-arch=sm_86`
- RTX 4090: `-arch=sm_89`

### 2. 运行测试

```bash
# 测试 CUDA 实现的正确性
python tests/test_cuda_mlp.py
```

测试会进行三方对比：
1. **CUDA 实现**（本项目）
2. **PyTorch 手动实现**（test_demo_mlp.py）
3. **PyTorch 自动求导**（ground truth）

### 3. Python 使用示例

```python
from tests.test_cuda_mlp import MLPClassificationCUDA
import torch

# 初始化
cuda_mlp = MLPClassificationCUDA()

# 准备数据
i, m, o = 3, 4, 3
x = torch.randn(i, 1, device='cuda')
y = torch.zeros(o, 1, device='cuda')
y[1] = 1.0  # 类别 1

w1 = torch.randn(m, i, device='cuda')
b1 = torch.randn(m, 1, device='cuda')
w2 = torch.randn(o, m, device='cuda')
b2 = torch.randn(o, 1, device='cuda')

# 前向传播
sigma, loss, intermediates = cuda_mlp.forward(
    w1, b1, w2, b2, x, y,
    save_intermediates=True,
    compute_loss=True
)

print(f"Loss: {loss:.6f}")
print(f"Predictions: {sigma.squeeze().tolist()}")

# 反向传播（仅一阶梯度）
grads, _ = cuda_mlp.backward(
    intermediates, w1, w2, x, y,
    compute_hessian=False
)

print(f"grad_w1: {grads['w1']}")

# 反向传播（一阶 + 二阶）
grads, hessians = cuda_mlp.backward(
    intermediates, w1, w2, x, y,
    compute_hessian=True
)

print(f"Hessian b2: {hessians['b2_full']}")
```

## 性能优化

### 已实现的优化

1. **数值稳定性**: Softmax 使用 `exp(x - max(x))` 避免溢出
2. **合并内存访问**: 使用 2D 线程块提高内存访问效率
3. **Warp Reduction**: 优化版本使用 warp shuffle 加速规约操作
4. **临时缓冲区管理**: 自动分配/释放中间变量

### 可选优化（未实现）

- 使用共享内存缓存权重矩阵
- 使用 cuBLAS 加速矩阵乘法
- Tensor Core 加速（需要 FP16/TF32）
- 批量处理（batch_size > 1）

## 数值验证结果

在测试网络（i=3, m=4, o=3）上的结果：

| 指标 | 差异 |
|------|------|
| 损失值 | < 1e-6 |
| 预测概率 | < 1e-6 |
| 梯度（w1, b1, w2, b2） | < 2e-7 |
| Hessian b2（完整） | < 2e-7 |
| Hessian b1（完整） | < 2e-7 |
| Hessian w2（对角） | < 1e-6 |
| Hessian w1（对角） | < 1e-6 |

所有差异都在单精度浮点数的数值精度范围内（~1e-7）。

## 理论背景

### Softmax + 交叉熵的优雅梯度

对于 softmax 输出 `σ = softmax(a2)` 和交叉熵损失 `L = -sum(y * log(σ))`，
损失对 logits 的梯度是：

```
dL/da2 = σ - y
```

这个简洁的形式来自于：
1. 交叉熵对 softmax 输出的导数：`dL/dσ_i = -y_i / σ_i`
2. Softmax 的雅可比矩阵：`dσ_i/da_j = σ_i * (δ_ij - σ_j)`
3. 链式法则：两者相乘后大量项消去，得到优雅的 `σ - y`

### Hessian 的半正定性

交叉熵损失的 Hessian `H_b2 = diag(σ) - σσ^T` 具有以下性质：
- **对称**: `H = H^T`
- **半正定**: 所有特征值 ≥ 0
- **零特征值**: 最小特征值为 0，对应特征向量为全 1 向量（softmax 的平移不变性）

这保证了牛顿法等二阶优化方法的稳定性。

## 开发笔记

- **作者**: Dy164
- **日期**: 2025-12-01
- **参考**: PyTorch manual implementation in `test_demo_mlp.py`

## 许可证

[根据项目需要填写]

## 相关资源

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
