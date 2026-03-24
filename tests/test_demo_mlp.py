from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Protocol, Iterable

from gsplat2ct.ops.data import Params, Data, Config


# ============================================================================
# 1. 输入数据结构
# ============================================================================
# 开发笔记 by Dy164 at 2025-12-01 15:45
# - 这里我们规范了计算模块的输入输出数据接口
# - 输入包括：可优化参数 Params、不可优化数据 Data、配置 Config
# - 三者的区别是，可优化参数参与计算且会被优化器修改
# - 不可优化数据参与计算但不会被修改
# - 配置是计算时的选项，不直接参与计算

class ParameterLike(Protocol):
    """抽象出一个“可优化参数”的最小接口。

    你后续的 NewtonParameter 可以实现这个 Protocol：
    - 有 .data: Tensor
    - 可以选择性有 .grad1, .hess 等字段

    开发笔记 by Dy164 at 2025-12-01 15:46
    - 后续我们可以把计算出来的梯度和局部海森都放在该参数下的 .grad1, .hess 字段中
    - 比如可以在优化器中定义一个注册梯度和海森的方法，完成上述操作
    - 之后直接就地使用，进行带阻尼的牛顿法优化
    - 优化器也管理梯度和海森的生命周期
    - 比如定义一个 clear_grad_hess() 方法，清除所有参数的梯度和海森
    - 这样在完成更新后可以直接就地消除梯度和海森
    """
    data: torch.Tensor


@dataclass
class Params:
    """可被优化的参数集合
    通用设计：用 name -> ParameterLike 的字典存储；
    对于具体模块（如 MLP），你可以在自己的子类中定义更具体的字段，
    然后在 __post_init__ 里同步到 param_map 里。
    """
    param_map: Dict[str, ParameterLike] = field(default_factory=dict)

    def items(self):
        return self.param_map.items()

    def values(self):
        return self.param_map.values()

    def keys(self):
        return self.param_map.keys()

    def get(self, name: str) -> Optional[ParameterLike]:
        return self.param_map.get(name)

    def add_param(self, name: str, param: ParameterLike) -> None:
        self.param_map[name] = param

    def __iter__(self) -> Iterable[ParameterLike]:
        return iter(self.param_map.values())











# ============================================================================
# 2. 输出数据结构
# ============================================================================

@dataclass
class ForwardResult:
    """前向计算结果
    
    outputs: 主要输出字典，通常包含 'loss' 等关键结果
    intermediates: 中间量字典，用于调试或后续反向传播
    """
    outputs: Dict[str, Any] = field(default_factory=dict)
    intermediates: Dict[str, Any] = field(default_factory=dict)
    
    def get_loss(self) -> torch.Tensor:
        """获取损失值的便捷方法"""
        return self.outputs.get('loss')


@dataclass
class GradResult:
    """一阶梯度结果"""
    param_grads: Dict[str, torch.Tensor] = field(default_factory=dict)

    def get(self, name: str) -> torch.Tensor:
        return self.param_grads[name]

    def set(self, name: str, grad: torch.Tensor) -> None:
        self.param_grads[name] = grad


@dataclass
class HessInfo:
    """某个参数的 Hessian 信息"""
    diag: Optional[torch.Tensor] = None
    full: Optional[torch.Tensor] = None
    approx: Optional[Any] = None


@dataclass
class HessResult:
    """二阶 Hessian 结果"""
    param_hess: Dict[str, HessInfo] = field(default_factory=dict)

    def get(self, name: str) -> HessInfo:
        return self.param_hess.get(name)

    def set(self, name: str, hess: HessInfo) -> None:
        self.param_hess[name] = hess


# ============================================================================
# 3. 抽象基类
# ============================================================================
# 开发笔记 by Dy164 at 2025-12-01 16:01
# - 定义一个抽象基类，规范二阶前向/反向计算的接口
# - 有三种不同的计算模式：仅前向、前向+一阶、前向+一阶+二阶
# - 仅前向对应验证或测试场景
# - 前向+一阶对应标准一阶训练场景
# - 前向+一阶+二阶对应二阶训练场景
# - 我们手写的算子最终都要封装到这个接口下
# - 封装时要重写这四个抽象方法

class SecondForwardBackwardBase(ABC):
    """二阶前向/反向计算的统一接口基类"""

    @abstractmethod
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """初始化过程"""
        raise NotImplementedError

    @abstractmethod
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """仅前向计算"""
        raise NotImplementedError

    @abstractmethod
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """前向 + 一阶反向计算"""
        raise NotImplementedError

    @abstractmethod
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """前向 + 一阶 + 二阶反向计算"""
        raise NotImplementedError


# ============================================================================
# 4. 手动实现
# ============================================================================
# 开发笔记 by Dy164 at 2025-12-01 16:04
# - gsplat框架下是将cuda计算结果和torch自动求导结果进行对比验证
# - 但是我认为应该要 cuda手动计算结果 + torch手动计算结果 + torch自动求导结果 三方对比
# - 这样更容易定位数值差异的来源
# - 后续编写规范化的pytest单元测试也应该遵循上述逻辑
# - 目前简单起见我们先比较 torch手动计算结果 + torch自动求导结果

class MLPManualForwardBackward(SecondForwardBackwardBase):
    """根据推导的公式手动计算梯度的 MLP 实现"""
    
    def __init__(self, i: int, m: int, o: int):
        self.i, self.m, self.o = i, m, o
        self.initialized = False
        # 缓存中间变量
        self.x = None
        self.y = None
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.a1 = None
        self.z1 = None
        self.a2 = None
        self.loss = None

    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """初始化：检查参数合法性"""
        # 检查必需的参数是否存在
        required_params = ['w1', 'b1', 'w2', 'b2']
        for name in required_params:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        
        self.initialized = True

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """仅前向计算"""
        # 提取参数和数据
        w1 = params.get('w1').data
        b1 = params.get('b1').data
        w2 = params.get('w2').data
        b2 = params.get('b2').data
        x = data.get('inputs')
        y = data.get('targets')
        task_type = data.get('task_type')
        
        # 前向计算
        a1 = torch.mm(w1, x) + b1
        z1 = torch.tanh(a1)
        a2 = torch.mm(w2, z1) + b2  # logits for classification
        
        # 根据任务类型计算损失
        if task_type == 'classification':
            # Softmax
            z_max = torch.max(a2, dim=0, keepdim=True)[0]  # 数值稳定
            exp_z = torch.exp(a2 - z_max)
            sigma = exp_z / torch.sum(exp_z, dim=0, keepdim=True)
            
            # 交叉熵损失
            loss = -torch.sum(y * torch.log(sigma + 1e-10))
            
            return ForwardResult(
                outputs={'loss': loss, 'predictions': sigma, 'logits': a2},
                intermediates={'a1': a1, 'z1': z1, 'a2': a2, 'sigma': sigma}
            )
        else:  # regression
            # L2 损失
            loss = 0.5 * torch.sum((a2 - y) ** 2)
            
            return ForwardResult(
                outputs={'loss': loss, 'predictions': a2},
                intermediates={'a1': a1, 'z1': z1, 'a2': a2}
            )

    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """前向 + 一阶反向计算"""
        # 提取参数和数据
        self.w1 = params.get('w1').data
        self.b1 = params.get('b1').data
        self.w2 = params.get('w2').data
        self.b2 = params.get('b2').data
        self.x = data.get('inputs')
        self.y = data.get('targets')
        self.task_type = data.get('task_type')
        
        # 前向计算
        self.a1 = torch.mm(self.w1, self.x) + self.b1
        self.z1 = torch.tanh(self.a1)
        self.a2 = torch.mm(self.w2, self.z1) + self.b2  # logits
        
        # 根据任务类型计算损失和梯度
        if self.task_type == 'classification':
            # Softmax
            z_max = torch.max(self.a2, dim=0, keepdim=True)[0]
            exp_z = torch.exp(self.a2 - z_max)
            self.sigma = exp_z / torch.sum(exp_z, dim=0, keepdim=True)
            
            # 交叉熵损失
            self.loss = -torch.sum(self.y * torch.log(self.sigma + 1e-10))
            
            # 前向结果
            fwd_result = ForwardResult(
                outputs={'loss': self.loss, 'predictions': self.sigma, 'logits': self.a2},
                intermediates={'a1': self.a1, 'z1': self.z1, 'a2': self.a2, 'sigma': self.sigma}
            )
            
            # 一阶反向计算（softmax + 交叉熵的优雅形式）
            # dL/dz = sigma - y
            error = self.sigma - self.y
            
        else:  # regression
            # L2 损失
            self.loss = 0.5 * torch.sum((self.a2 - self.y) ** 2)
            
            # 前向结果
            fwd_result = ForwardResult(
                outputs={'loss': self.loss, 'predictions': self.a2},
                intermediates={'a1': self.a1, 'z1': self.z1, 'a2': self.a2}
            )
            
            # 一阶反向计算（L2 损失）
            error = self.a2 - self.y
        
        # 对第二层参数的梯度（两种任务相同）
        grad_w2 = torch.mm(error, self.z1.t())
        grad_b2 = error.clone()
        
        # 对第一层参数的梯度（两种任务相同）
        s = torch.mm(self.w2.t(), error)
        d = 1 - self.z1 ** 2
        grad_a1 = s * d
        grad_w1 = torch.mm(grad_a1, self.x.t())
        grad_b1 = grad_a1.clone()
        
        # 梯度结果
        grad_result = GradResult()
        grad_result.set('w1', grad_w1)
        grad_result.set('b1', grad_b1)
        grad_result.set('w2', grad_w2)
        grad_result.set('b2', grad_b2)
        
        return fwd_result, grad_result

    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """前向 + 一阶 + 二阶反向计算"""
        # 先计算前向和一阶
        fwd_result, grad_result = self.forward_backward1(params, data, config)
        
        # 二阶计算
        if self.task_type == 'classification':
            # Softmax + 交叉熵的 Hessian
            # H = diag(sigma) - sigma * sigma^T
            sigma_vec = self.sigma.squeeze()  # (o,)
            hess_a2 = torch.diag(sigma_vec) - torch.mm(self.sigma, self.sigma.t())
            
            # 对 b2 的二阶导数（与 a2 相同，因为 da2/db2 = I）
            hess_b2 = hess_a2.clone()
            
            # 对 w2 的二阶导数（Kronecker积）
            zzT = torch.mm(self.z1, self.z1.t())
            hess_w2 = torch.kron(hess_a2, zzT)
            
            # 反向传播到第一层
            # 使用链式法则: d²L/da1² = (dz1/da1)^T * W2^T * H_a2 * W2 * (dz1/da1) + (dL/dz2) * d²z2/dz1da1
            # 其中 dz1/da1 = diag(1 - z1^2), d²z1/da1² = diag(-2*z1*(1-z1^2))
            d = 1 - self.z1 ** 2  # tanh的一阶导数
            D = torch.diag(d.squeeze())
            
            # 第一项：二阶传播项
            # D * (W2^T * H_a2 * W2) * D
            w2T_h_w2 = torch.mm(torch.mm(self.w2.t(), hess_a2), self.w2)
            first_term = torch.mm(torch.mm(D, w2T_h_w2), D)
            
            # 第二项：一阶导数 * tanh的二阶导数
            # (dL/dz2) * d²z1/da1² 其中 dL/dz2 = sigma - y
            error = self.sigma - self.y
            s = torch.mm(self.w2.t(), error)  # W2^T * (sigma - y)
            second_term = -2 * torch.diag((s * self.z1 * d).squeeze())
            
            hess_a1 = first_term + second_term
            
        else:  # regression (L2 loss)
            error = self.a2 - self.y
            s = torch.mm(self.w2.t(), error)
            d = 1 - self.z1 ** 2
            
            # 对 b2 的二阶导数（恒等矩阵）
            hess_b2 = torch.eye(self.o, self.o)
            
            # 对 w2 的二阶导数（Kronecker积）
            zzT = torch.mm(self.z1, self.z1.t())
            hess_w2 = torch.kron(torch.eye(self.o, self.o), zzT)
            
            # 对 a1 的二阶导数（Hessian矩阵）
            w2T_w2 = torch.mm(self.w2.t(), self.w2)
            D = torch.diag(d.squeeze())
            first_term = torch.mm(torch.mm(D, w2T_w2), D)
            s_zd = s * self.z1 * d
            diag_s_zd = torch.diag(s_zd.squeeze())
            hess_a1 = first_term - 2 * diag_s_zd
        
        # 对 b1 的二阶导数（与 a1 相同）
        hess_b1 = hess_a1.clone()
        
        # 对 w1 的二阶导数（Kronecker积）
        xxT = torch.mm(self.x, self.x.t())
        hess_w1 = torch.kron(hess_a1, xxT)
        
        # Hessian 结果
        hess_result = HessResult()
        hess_result.set('w1', hess_w1)
        hess_result.set('b1', hess_b1)
        hess_result.set('w2', hess_w2)
        hess_result.set('b2', hess_b2)
        
        return fwd_result, grad_result, hess_result


# ============================================================================
# 5. 自动求导实现
# ============================================================================
# 开发笔记 by Dy164 at 2025-12-01 16:06
# - 目前我们调用 PyTorch 的 autograd 来计算梯度
# - 调用 torch.autograd.functional.hessian 来计算 Hessian
# - 需要格外注意Hessian的大小，小心测试数据过大导致内存爆炸

class SimpleNetwork(nn.Module):
    """简单的两层网络"""
    
    def __init__(self, i, m, o, task_type='regression'):
        super(SimpleNetwork, self).__init__()
        self.linear1 = nn.Linear(i, m, bias=True)
        self.linear2 = nn.Linear(m, o, bias=True)
        self.task_type = task_type
        
    def forward(self, x):
        # x的形状应该是(batch_size, input_size)
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        return x  # 返回 logits（分类）或直接输出（回归）


class MLPAutoForwardBackward(SecondForwardBackwardBase):
    """使用PyTorch自动求导计算梯度的 MLP 实现"""
    
    def __init__(self, i: int, m: int, o: int, task_type: str = 'regression'):
        self.i, self.m, self.o = i, m, o
        self.task_type = task_type
        self.model = SimpleNetwork(i, m, o, task_type)
        self.initialized = False
        self.x = None
        self.y = None

    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """初始化：检查参数合法性"""
        # 检查必需的参数是否存在
        required_params = ['w1', 'b1', 'w2', 'b2']
        for name in required_params:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        
        self.initialized = True

    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """仅前向计算"""
        # 提取参数和数据
        w1 = params.get('w1').data
        b1 = params.get('b1').data
        w2 = params.get('w2').data
        b2 = params.get('b2').data
        x = data.get('inputs')
        y = data.get('targets')
        task_type = data.get('task_type')
        
        # 将参数设置到模型中
        with torch.no_grad():
            self.model.linear1.weight.copy_(w1)
            self.model.linear1.bias.copy_(b1.squeeze())
            self.model.linear2.weight.copy_(w2)
            self.model.linear2.bias.copy_(b2.squeeze())
        
        # 前向传播
        logits = self.model(x.t())  # (batch_size=1, output_size)
        
        if task_type == 'classification':
            # Softmax + 交叉熵
            sigma = torch.softmax(logits, dim=1)
            loss = -torch.sum(y.t() * torch.log(sigma + 1e-10))
            
            return ForwardResult(
                outputs={'loss': loss, 'predictions': sigma.t(), 'logits': logits.t()},
                intermediates={}
            )
        else:  # regression
            # L2 损失
            loss = 0.5 * torch.sum((logits - y.t()) ** 2)
            
            return ForwardResult(
                outputs={'loss': loss, 'predictions': logits.t()},
                intermediates={}
            )

    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult]:
        """前向 + 一阶反向计算"""
        # 保存输入和目标，供后续梯度计算使用
        self.x = data.get('inputs')
        self.y = data.get('targets')
        task_type = data.get('task_type')
        
        # 提取参数
        w1 = params.get('w1').data
        b1 = params.get('b1').data
        w2 = params.get('w2').data
        b2 = params.get('b2').data
        
        # 将参数设置到模型中
        with torch.no_grad():
            self.model.linear1.weight.copy_(w1)
            self.model.linear1.bias.copy_(b1.squeeze())
            self.model.linear2.weight.copy_(w2)
            self.model.linear2.bias.copy_(b2.squeeze())
        
        # 确保梯度被清零
        self.model.zero_grad()
        
        # 前向传播
        logits = self.model(self.x.t())
        
        if task_type == 'classification':
            # Softmax + 交叉熵
            sigma = torch.softmax(logits, dim=1)
            loss = -torch.sum(self.y.t() * torch.log(sigma + 1e-10))
            
            # 前向结果
            fwd_result = ForwardResult(
                outputs={'loss': loss, 'predictions': sigma.t(), 'logits': logits.t()},
                intermediates={}
            )
        else:  # regression
            # L2 损失
            loss = 0.5 * torch.sum((logits - self.y.t()) ** 2)
            
            # 前向结果
            fwd_result = ForwardResult(
                outputs={'loss': loss, 'predictions': logits.t()},
                intermediates={}
            )
        
        # 反向传播
        loss.backward()
        
        # 提取梯度
        grad_result = GradResult()
        grad_result.set('w1', self.model.linear1.weight.grad.clone())
        grad_result.set('b1', self.model.linear1.bias.grad.unsqueeze(1).clone())
        grad_result.set('w2', self.model.linear2.weight.grad.clone())
        grad_result.set('b2', self.model.linear2.bias.grad.unsqueeze(1).clone())
        
        # 清除梯度
        self.model.zero_grad()
        
        return fwd_result, grad_result

    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> tuple[ForwardResult, GradResult, HessResult]:
        """前向 + 一阶 + 二阶反向计算"""
        # 先计算前向和一阶
        fwd_result, grad_result = self.forward_backward1(params, data, config)
        
        # 计算二阶 Hessian
        hess_result = self._compute_hessians_autograd()
        
        return fwd_result, grad_result, hess_result

    def _compute_hessians_autograd(self):
        """用扁平参数直接实现前向，计算完整 Hessian，并截取 w1/b1/w2/b2 四个对角块"""
        # 获取当前参数
        w1 = self.model.linear1.weight
        b1 = self.model.linear1.bias
        w2 = self.model.linear2.weight
        b2 = self.model.linear2.bias
        
        m, i = w1.shape
        o, _ = w2.shape
        
        # 扁平化参数
        flat_w1 = w1.reshape(-1)
        flat_b1 = b1.reshape(-1)
        flat_w2 = w2.reshape(-1)
        flat_b2 = b2.reshape(-1)
        
        flat_params = torch.cat([flat_w1, flat_b1, flat_w2, flat_b2])
        
        sizes = {
            'w1': flat_w1.numel(),
            'b1': flat_b1.numel(),
            'w2': flat_w2.numel(),
            'b2': flat_b2.numel(),
        }
        
        x = self.x
        y = self.y
        task_type = self.task_type
        
        def loss_from_flat(flat):
            # 根据扁平参数拆出 w1, b1, w2, b2，然后用它们手写前向 + loss
            offset = 0
            
            num_w1 = sizes['w1']
            num_b1 = sizes['b1']
            num_w2 = sizes['w2']
            num_b2 = sizes['b2']
            
            w1_flat = flat[offset: offset + num_w1]
            offset += num_w1
            b1_flat = flat[offset: offset + num_b1]
            offset += num_b1
            w2_flat = flat[offset: offset + num_w2]
            offset += num_w2
            b2_flat = flat[offset: offset + num_b2]
            
            W1 = w1_flat.view(m, i)
            b1_vec = b1_flat.view(m, 1)
            W2 = w2_flat.view(o, m)
            b2_vec = b2_flat.view(o, 1)
            
            a1 = torch.mm(W1, x) + b1_vec
            z1 = torch.tanh(a1)
            a2 = torch.mm(W2, z1) + b2_vec
            
            if task_type == 'classification':
                # Softmax + 交叉熵
                z_max = torch.max(a2, dim=0, keepdim=True)[0]
                exp_z = torch.exp(a2 - z_max)
                sigma = exp_z / torch.sum(exp_z, dim=0, keepdim=True)
                loss = -torch.sum(y * torch.log(sigma + 1e-10))
            else:  # regression
                loss = 0.5 * torch.sum((a2 - y) ** 2)
            
            return loss
        
        # 计算完整Hessian
        H_full = torch.autograd.functional.hessian(loss_from_flat, flat_params)
        
        # 截取对角块
        hess_result = HessResult()
        offset = 0
        for name in ['w1', 'b1', 'w2', 'b2']:
            n = sizes[name]
            H_block = H_full[offset: offset + n, offset: offset + n]
            hess_result.set(name, H_block)
            offset += n
        
        return hess_result


# ============================================================================
# 6. 主函数：测试和比较
# ============================================================================
# 开发笔记 by Dy164 at 2025-12-01 16:09
# - 这只是一个简单实现
# - 后续应该按照pytest规范编写单元测试

def test_regression():
    """测试回归任务（L2损失）"""
    print("\n" + "=" * 80)
    print("测试 1: 回归任务（L2 损失）")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    i, m, o = 3, 4, 2
    x = torch.randn(i, 1, requires_grad=False)
    y = torch.randn(o, 1, requires_grad=False)
    w1 = torch.randn(m, i, requires_grad=True)
    b1 = torch.randn(m, 1, requires_grad=True)
    w2 = torch.randn(o, m, requires_grad=True)
    b2 = torch.randn(o, 1, requires_grad=True)
    
    params = Params()
    params.add_param('w1', type('Param', (), {'data': w1})())
    params.add_param('b1', type('Param', (), {'data': b1})())
    params.add_param('w2', type('Param', (), {'data': w2})())
    params.add_param('b2', type('Param', (), {'data': b2})())
    
    data = Data()
    data.set('inputs', x)
    data.set('targets', y)
    data.set('task_type', 'regression')
    
    config = Config()
    config.set('hessian_type', 'full')
    
    # 手动实现
    manual_fb = MLPManualForwardBackward(i, m, o)
    manual_fb.initialize(params, config)
    fwd_manual, grad_manual, hess_manual = manual_fb.forward_backward2(params, data, config)
    
    # 自动求导
    auto_fb = MLPAutoForwardBackward(i, m, o, task_type='regression')
    auto_fb.initialize(params, config)
    fwd_auto, grad_auto, hess_auto = auto_fb.forward_backward2(params, data, config)
    
    # 比较结果
    print(f"损失值 - 手动: {fwd_manual.get_loss().item():.6f}, 自动: {fwd_auto.get_loss().item():.6f}")
    print(f"损失差异: {abs(fwd_manual.get_loss().item() - fwd_auto.get_loss().item()):.6e}")
    
    for name in ['w1', 'b1', 'w2', 'b2']:
        grad_diff = torch.norm(grad_manual.get(name) - grad_auto.get(name))
        print(f"梯度 {name} 差异: {grad_diff.item():.6e}")
        
        hess_m = hess_manual.get(name)
        hess_a = hess_auto.get(name)
        hess_diff = torch.norm(hess_m - hess_a)
        print(f"Hessian {name} 差异: {hess_diff.item():.6e}")


def test_classification():
    """测试分类任务（Softmax + 交叉熵）"""
    print("\n" + "=" * 80)
    print("测试 2: 分类任务（Softmax + 交叉熵损失）")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(123)
    np.random.seed(123)
    
    i, m, o = 3, 4, 3  # 3类分类
    x = torch.randn(i, 1, requires_grad=False)
    
    # 生成 one-hot 标签
    true_class = 1
    y = torch.zeros(o, 1)
    y[true_class] = 1.0
    
    w1 = torch.randn(m, i, requires_grad=True)
    b1 = torch.randn(m, 1, requires_grad=True)
    w2 = torch.randn(o, m, requires_grad=True)
    b2 = torch.randn(o, 1, requires_grad=True)
    
    print(f"网络尺寸: 输入={i}, 隐藏层={m}, 输出={o}")
    print(f"真实类别: {true_class}")
    print(f"one-hot标签: {y.squeeze().tolist()}")
    print()
    
    params = Params()
    params.add_param('w1', type('Param', (), {'data': w1})())
    params.add_param('b1', type('Param', (), {'data': b1})())
    params.add_param('w2', type('Param', (), {'data': w2})())
    params.add_param('b2', type('Param', (), {'data': b2})())
    
    data = Data()
    data.set('inputs', x)
    data.set('targets', y)
    data.set('task_type', 'classification')
    
    config = Config()
    config.set('hessian_type', 'full')
    
    # 手动实现
    print("手动计算（根据推导公式）:")
    print("-" * 80)
    manual_fb = MLPManualForwardBackward(i, m, o)
    manual_fb.initialize(params, config)
    fwd_manual, grad_manual, hess_manual = manual_fb.forward_backward2(params, data, config)
    
    print(f"损失值: {fwd_manual.get_loss().item():.6f}")
    print(f"预测概率: {fwd_manual.outputs['predictions'].squeeze().tolist()}")
    print(f"预测类别: {torch.argmax(fwd_manual.outputs['predictions']).item()}")
    print()
    
    # 自动求导
    print("自动求导（PyTorch autograd）:")
    print("-" * 80)
    auto_fb = MLPAutoForwardBackward(i, m, o, task_type='classification')
    auto_fb.initialize(params, config)
    fwd_auto, grad_auto, hess_auto = auto_fb.forward_backward2(params, data, config)
    
    print(f"损失值: {fwd_auto.get_loss().item():.6f}")
    print(f"预测概率: {fwd_auto.outputs['predictions'].squeeze().tolist()}")
    print(f"预测类别: {torch.argmax(fwd_auto.outputs['predictions']).item()}")
    print()
    
    # 详细比较
    print("=" * 80)
    print("详细结果比较")
    print("=" * 80)
    
    # 损失值
    loss_diff = abs(fwd_manual.get_loss().item() - fwd_auto.get_loss().item())
    print(f"损失值差异: {loss_diff:.6e}")
    
    # 预测概率
    pred_diff = torch.norm(fwd_manual.outputs['predictions'] - fwd_auto.outputs['predictions'])
    print(f"预测概率差异: {pred_diff.item():.6e}")
    print()
    
    # 一阶梯度
    print("一阶梯度比较:")
    for name in ['w1', 'b1', 'w2', 'b2']:
        grad_m = grad_manual.get(name)
        grad_a = grad_auto.get(name)
        diff = torch.norm(grad_m - grad_a)
        rel_diff = diff / (torch.norm(grad_m) + 1e-8)
        print(f"  {name}: 差异={diff.item():.6e}, 相对差异={rel_diff.item():.6e}")
    print()
    
    # 二阶 Hessian
    print("二阶 Hessian 比较:")
    for name in ['w1', 'b1', 'w2', 'b2']:
        hess_m = hess_manual.get(name)
        hess_a = hess_auto.get(name)
        diff = torch.norm(hess_m - hess_a)
        rel_diff = diff / (torch.norm(hess_m) + 1e-8)
        print(f"  {name}: 差异={diff.item():.6e}, 相对差异={rel_diff.item():.6e}")
    print()
    
    # 打印详细数值（w2和b2的梯度和Hessian）
    print("=" * 80)
    print("详细数值：w2 和 b2 的梯度和 Hessian")
    print("=" * 80)
    
    print("\nw2 梯度（手动）:")
    print(grad_manual.get('w2'))
    print("w2 梯度（自动）:")
    print(grad_auto.get('w2'))
    
    print("\nb2 梯度（手动）:")
    print(grad_manual.get('b2').squeeze())
    print("b2 梯度（自动）:")
    print(grad_auto.get('b2').squeeze())
    
    print("\nb2 Hessian（手动）:")
    print(hess_manual.get('b2'))
    print("b2 Hessian（自动）:")
    print(hess_auto.get('b2'))
    
    # 验证 Hessian 的性质
    print("\n" + "=" * 80)
    print("验证 Hessian 矩阵的性质")
    print("=" * 80)
    
    H = hess_manual.get('b2')
    print(f"b2 Hessian 形状: {H.shape}")
    print(f"对称性检查 (||H - H^T||): {torch.norm(H - H.t()).item():.6e}")
    
    # 特征值分析
    eigenvalues = torch.linalg.eigvals(H).real
    print(f"特征值: {eigenvalues.tolist()}")
    print(f"最小特征值: {eigenvalues.min().item():.6e}")
    print(f"是否半正定: {(eigenvalues >= -1e-6).all().item()}")
    
    # 验证零特征值对应的特征向量
    eigvals, eigvecs = torch.linalg.eig(H)
    eigvals = eigvals.real
    min_idx = torch.argmin(torch.abs(eigvals))
    print(f"最小特征值对应的特征向量: {eigvecs[:, min_idx].real.tolist()}")
    print("(应该接近全1向量，对应softmax的平移不变性)")


def main():
    print("=" * 80)
    print("MLP 二阶前向/反向计算测试")
    print("=" * 80)
    print("包含两种任务:")
    print("1. 回归任务 (L2 损失)")
    print("2. 分类任务 (Softmax + 交叉熵损失)")
    print("=" * 80)
    
    # 测试回归任务
    test_regression()
    
    # 测试分类任务
    test_classification()


if __name__ == "__main__":
    main()
