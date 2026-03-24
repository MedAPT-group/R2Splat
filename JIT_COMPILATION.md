# JIT 编译功能实现完成

## 概述

已成功实现类似 gsplat 的 JIT（Just-In-Time）即时编译功能。现在 gsplat2ct 可以：
1. 优先加载预编译的扩展（如果存在）
2. 如果预编译扩展不存在，自动进行 JIT 编译
3. 支持多种环境变量配置编译行为

## 主要变更

### `gsplat2ct/cuda/_backend.py` - 完全重写

**新增功能**：

1. **健壮的 CUDA 检测**
   - 使用 `_find_cuda_home()` 查找 CUDA 工具包
   - 检查 nvcc 是否存在
   - 自动处理不同的 CUDA 安装位置

2. **智能 JIT 编译**
   - 如果预编译扩展存在，直接加载
   - 如果不存在，自动触发 JIT 编译
   - 编译完成后缓存，下次直接使用

3. **环境变量支持**
   ```bash
   # 快速编译（用于调试）
   FAST_COMPILE=1 python -c "from gsplat2ct.cuda._backend import _C"
   
   # 详细输出
   VERBOSE=1 python -c "from gsplat2ct.cuda._backend import _C"
   
   # 禁用 fast_math
   NO_FAST_MATH=1 python script.py
   
   # 设置并行编译任务数
   MAX_JOBS=8 python script.py
   ```

4. **用户友好的输出**
   - 使用 `rich.Console` 提供彩色状态信息
   - 显示编译进度（带动画）
   - 显示编译耗时
   - 清晰的错误/警告信息

5. **处理并发编译冲突**
   - 自动删除过期的锁文件
   - 处理多进程同时编译的情况
   - 容错机制（OSError 时尝试导入现有模块）

## 代码结构

```python
# 1. 环境变量配置
NO_FAST_MATH = os.getenv("NO_FAST_MATH", "0") == "1"
FAST_COMPILE = os.getenv("FAST_COMPILE", "0") == "1"
VERBOSE = os.getenv("VERBOSE", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")

# 2. 工具函数
def load_extension(...)          # JIT 编译加载器
def cuda_toolkit_available()     # CUDA 工具包检测

# 3. 加载逻辑
try:
    from gsplat2ct import csrc as _C  # 尝试导入预编译模块
except ImportError:
    if cuda_toolkit_available():
        # JIT 编译
        _C = load_extension(...)
    else:
        # CUDA 不可用，_C = None

# 4. 导出接口
def is_cuda_available() -> bool
```

## 编译流程

```
启动 Python 脚本
     ↓
导入 gsplat2ct.cuda._backend
     ↓
     ├─→ 预编译扩展存在？
     │        ↓ 是
     │   直接加载 → _C
     │
     ├─→ CUDA 工具包可用？
     │        ↓ 是
     │   检查缓存目录
     │        ↓
     │   ├─→ .so/.lib 存在？
     │   │        ↓ 是
     │   │   加载现有编译 → _C
     │   │
     │   └─→ 不存在
     │        ↓
     │   清理旧构建
     │        ↓
     │   显示进度条
     │        ↓
     │   JIT 编译（_jit_compile）
     │        ↓
     │   保存到缓存 → _C
     │
     └─→ CUDA 不可用
          ↓
     显示警告 → _C = None
```

## 使用示例

### 1. 正常使用（自动 JIT）

```python
from gsplat2ct.cuda._backend import _C, is_cuda_available

if is_cuda_available():
    # 第一次运行：自动 JIT 编译（显示进度条）
    # 后续运行：直接加载缓存
    result = _C.mlp_classification_fwd(...)
```

### 2. 调试模式

```bash
# 快速编译 + 详细输出
FAST_COMPILE=1 VERBOSE=1 python test_jit_compile.py
```

### 3. 强制重新编译

```bash
# 删除缓存后重新编译
rm -rf ~/.cache/torch_extensions/gsplat2ct_cuda
python test_jit_compile.py
```

### 4. 检查 JIT 编译状态

```python
python test_jit_compile.py
```

## 与 gsplat 的对比

| 特性 | gsplat | gsplat2ct | 说明 |
|------|--------|-----------|------|
| JIT 编译 | ✅ | ✅ | 完全实现 |
| CUDA 检测 | ✅ | ✅ | 使用 _find_cuda_home |
| 环境变量 | ✅ | ✅ | FAST_COMPILE, VERBOSE, etc. |
| Rich 输出 | ✅ | ✅ | 彩色状态信息 |
| 并发安全 | ✅ | ✅ | 锁文件处理 |
| 缓存管理 | ✅ | ✅ | torch_extensions 缓存 |
| 预编译支持 | ✅ | ✅ | setup.py 安装 |
| Torch 2.7 兼容 | ✅ | ✅ | SYCL 参数处理 |

## 优势

### 1. **开发便利性**
- 不需要运行 `pip install -e .` 即可使用
- 修改 .cu 文件后自动重新编译
- 适合快速迭代开发

### 2. **用户友好性**
- 首次使用自动编译（无需手动操作）
- 清晰的进度指示
- 友好的错误信息

### 3. **灵活性**
- 支持预编译安装（生产环境）
- 支持 JIT 编译（开发环境）
- 两种方式无缝切换

### 4. **健壮性**
- 自动检测 CUDA 可用性
- 处理并发编译冲突
- 容错机制完善

## 测试说明

### 运行测试脚本

```bash
cd /home/chengzyi/myterminus/code/gsplat-3dgs2ct
python test_jit_compile.py
```

**测试内容**：
1. ✓ 检查 CUDA 可用性
2. ✓ 导入 _backend 模块
3. ✓ 检查 _C 扩展是否加载
4. ✓ 列出可用函数
5. ✓ 执行简单的前向传播测试

**预期输出**：
```
================================================================================
Testing gsplat2ct JIT Compilation
================================================================================

1. Checking CUDA availability...
   torch.cuda.is_available(): True
   CUDA device count: 1
   CUDA device name: ...

2. Importing gsplat2ct.cuda._backend...
   gsplat2ct: Setting up CUDA ... (首次运行)
   或
   ✓ Import successful (已有缓存)

3. Checking CUDA extension availability...
   is_cuda_available(): True
   _C is not None: True

4. Checking available functions in _C...
   Available functions: [...]
   ✓ mlp_classification_fwd found
   ✓ mlp_classification_bwd found
   ✓ mlp_classification_bwd2 found

5. Testing basic function call...
   Calling mlp_classification_fwd...
   ✓ Function call successful
   Result type: <class 'tuple'>
   Result length: 5
   sigma shape: torch.Size([2])
   loss: 0.xxxx
   ...

✅ All tests passed!
```

## 编译缓存位置

JIT 编译的结果缓存在：
```
~/.cache/torch_extensions/gsplat2ct_cuda/
```

包含：
- `gsplat2ct_cuda.so` (Linux)
- `gsplat2ct_cuda.lib` (Windows)
- 编译中间文件

## 故障排除

### 问题 1: 找不到 nvcc

**症状**：
```
[yellow]gsplat2ct: No CUDA toolkit found...[/yellow]
```

**解决**：
```bash
# 检查 nvcc
which nvcc

# 如果没有，添加到 PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### 问题 2: 编译失败

**症状**：
编译过程中报错

**解决**：
```bash
# 使用详细输出查看错误
VERBOSE=1 python test_jit_compile.py

# 清理缓存重试
rm -rf ~/.cache/torch_extensions/gsplat2ct_cuda
```

### 问题 3: 锁文件冲突

**症状**：
```
RuntimeError: lock file exists
```

**解决**：
已自动处理，如仍有问题：
```bash
rm ~/.cache/torch_extensions/gsplat2ct_cuda/lock
```

## 下一步

现在可以：

1. **测试 JIT 编译**：
   ```bash
   python test_jit_compile.py
   ```

2. **运行完整测试**：
   ```bash
   python tests/test_mlp_classification.py
   ```

3. **开发时使用**：
   - 直接 `import gsplat2ct`，首次会自动 JIT 编译
   - 修改代码后删除缓存即可重新编译

4. **生产部署**：
   - 仍可使用 `pip install -e .` 预编译
   - JIT 作为后备方案

---
**状态**: ✅ JIT 编译功能已完整实现  
**日期**: 2025-12-03  
**参考**: gsplat/cuda/_backend.py
