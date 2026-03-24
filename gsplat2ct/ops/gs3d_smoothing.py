"""
3D Gaussian Splatting with Total Variation Smoothing Loss

This module provides end-to-end implementations for 3D Gaussian Splatting with TV regularization:
- Forward: Render 3D Gaussians to voxel volume, compute TV smoothing loss
- Backward1: Compute first-order gradients w.r.t. Gaussian parameters
- Backward2: Compute second-order Hessians w.r.t. Gaussian parameters

The pipeline is:
    3D Gaussian Parameters (means3d, conics, opacities)
        -> Voxelization -> Rendered Volume (dim0, dim1, dim2)
        -> TV Loss -> Scalar Loss

For second-order optimization:
    Loss -> v_volume (first derivative) + h_volume (second derivative)
         -> Chain rule through voxelization
         -> Gradients + Hessians for Gaussian parameters

Note on TV Loss:
    - TV loss is L1-based: TV = mean(|∇volume|)
    - Hessian is zero almost everywhere (except at zero crossings)
    - For second-order, we use the Gauss-Newton approximation: H ≈ J^T @ J

Author: Dy164
Date: 2026-02-03
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


# ============================================================================
# Data Structures for 3D Gaussian Splatting with Smoothing
# ============================================================================

@dataclass
class GS3DSmoothingParams(Params):
    """Parameters for 3D Gaussian Splatting with Smoothing.
    
    Attributes:
        means3d: 3D Gaussian centers (N, 3), format: (x, y, z)
        conics: Inverse covariance matrices (N, 6), stored as [a, p, q, b, r, c]
        opacities: Gaussian opacities (N,)
    """
    means3d: Optional[torch.Tensor] = None
    conics: Optional[torch.Tensor] = None
    opacities: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        super().__init__()
        if self.means3d is not None:
            self.add_param('means3d', _ParamWrapper(self.means3d))
        if self.conics is not None:
            self.add_param('conics', _ParamWrapper(self.conics))
        if self.opacities is not None:
            self.add_param('opacities', _ParamWrapper(self.opacities))


@dataclass
class GS3DSmoothingData(Data):
    """Data for 3D Gaussian Splatting with Smoothing.
    
    Attributes:
        radii: Gaussian radii for tile intersection (N, 3) or (N,)
        dim0: Volume depth (z dimension)
        dim1: Volume height (y dimension)
        dim2: Volume width (x dimension)
        tile_size: Tile size for voxelization (default 4)
    """
    radii: Optional[torch.Tensor] = None
    dim0: int = 16
    dim1: int = 16
    dim2: int = 16
    tile_size: int = 4
    
    def __post_init__(self):
        super().__init__()
        if self.radii is not None:
            self.set('radii', self.radii)
        self.set('dim0', self.dim0)
        self.set('dim1', self.dim1)
        self.set('dim2', self.dim2)
        self.set('tile_size', self.tile_size)


@dataclass
class GS3DSmoothingConfig(Config):
    """Configuration for 3D Gaussian Splatting with Smoothing.
    
    Attributes:
        tile_size: Size of each tile in voxels (default 4)
        tv_weight: Weight for TV loss (default 1.0)
    """
    tile_size: int = 4
    tv_weight: float = 1.0
    
    def __post_init__(self):
        super().__init__()


class _ParamWrapper:
    """Simple wrapper to hold parameter data."""
    def __init__(self, data: torch.Tensor):
        self.data = data


# ============================================================================
# Helper: TV Loss computation
# ============================================================================

def compute_tv_loss(volume: torch.Tensor, smooth: bool = False, eps: float = 1e-4) -> torch.Tensor:
    """
    Compute Total Variation loss for a 3D volume.
    
    TV = mean(|∂x| + |∂y| + |∂z|) per voxel
    
    This matches the CUDA implementation where each voxel contributes
    the sum of its forward differences in all 3 dimensions.
    
    Args:
        volume: (D, H, W) voxel volume
        smooth: If True, use smooth L1 (sqrt(x^2 + eps)) instead of |x|
                This makes Hessian non-zero and matches CUDA bwd2 behavior.
        eps: Smoothing parameter for smooth L1 (default 1e-4)
        
    Returns:
        Scalar TV loss
    """
    D, H, W = volume.shape
    
    # Compute differences along each axis (forward differences)
    diff_x = volume[1:, :, :] - volume[:-1, :, :]  # (D-1, H, W)
    diff_y = volume[:, 1:, :] - volume[:, :-1, :]  # (D, H-1, W)
    diff_z = volume[:, :, 1:] - volume[:, :, :-1]  # (D, H, W-1)
    
    if smooth:
        # Smooth L1: sqrt(x^2 + eps) - matches CUDA bwd2
        dx = torch.sqrt(diff_x * diff_x + eps)
        dy = torch.sqrt(diff_y * diff_y + eps)
        dz = torch.sqrt(diff_z * diff_z + eps)
    else:
        # Pure L1: |x|
        dx = torch.abs(diff_x)
        dy = torch.abs(diff_y)
        dz = torch.abs(diff_z)
    
    # Create TV map: for each voxel, sum of its forward differences
    # Initialize with zeros
    tv_map = torch.zeros_like(volume)
    
    # Add contributions from each direction
    # For d dimension: voxels [0:D-1] have forward diff
    tv_map[:-1, :, :] += dx
    
    # For h dimension: voxels [:, 0:H-1, :] have forward diff
    tv_map[:, :-1, :] += dy
    
    # For w dimension: voxels [:, :, 0:W-1] have forward diff
    tv_map[:, :, :-1] += dz
    
    # Average over all voxels (matching CUDA)
    return tv_map.sum() / (D * H * W)


def compute_tv_gradient(volume: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of TV loss w.r.t. volume.
    
    For TV = mean(sum of |forward_diffs| per voxel),
    ∂TV/∂v[i,j,k] includes contributions from:
    - Forward diff from this voxel: sign(v[i+1,j,k] - v[i,j,k])
    - Backward diff to this voxel: -sign(v[i,j,k] - v[i-1,j,k])
    
    This matches the CUDA backward implementation.
    
    Args:
        volume: (D, H, W) voxel volume
        
    Returns:
        (D, H, W) gradient tensor
    """
    D, H, W = volume.shape
    grad = torch.zeros_like(volume)
    
    # Normalization factor (same as forward: 1 / total_voxels)
    norm = 1.0 / (D * H * W)
    
    # Along dim0 (D)
    diff_d = volume[1:, :, :] - volume[:-1, :, :]
    sign_d = torch.sign(diff_d)
    # Forward contribution: v[d, h, w] -> v[d+1, h, w]
    grad[:-1, :, :] -= sign_d * norm
    # Backward contribution: v[d+1, h, w] <- v[d, h, w]
    grad[1:, :, :] += sign_d * norm
    
    # Along dim1 (H)
    diff_h = volume[:, 1:, :] - volume[:, :-1, :]
    sign_h = torch.sign(diff_h)
    grad[:, :-1, :] -= sign_h * norm
    grad[:, 1:, :] += sign_h * norm
    
    # Along dim2 (W)
    diff_w = volume[:, :, 1:] - volume[:, :, :-1]
    sign_w = torch.sign(diff_w)
    grad[:, :, :-1] -= sign_w * norm
    grad[:, :, 1:] += sign_w * norm
    
    return grad


# ============================================================================
# Autograd Implementation (for testing/validation)
# ============================================================================

class GS3DSmoothingAutograd(SecondForwardBackwardBase):
    """
    3D Gaussian Splatting with TV Smoothing using torch autograd.
    
    This implementation uses pure PyTorch operations and autograd for:
    - Forward voxelization + TV loss
    - First-order gradients (autograd)
    - Second-order Hessians (double autograd)
    
    Use this for testing and validation against the CUDA implementation.
    """
    
    def __init__(
        self,
        tv_weight: float = 1.0,
        tile_size: int = 4,
        eps: float = 1e-6,
    ):
        self.tv_weight = tv_weight
        self.tile_size = tile_size
        self.eps = eps
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        self.initialized = True
    
    def _voxelize_torch(
        self,
        means3d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        dim0: int,
        dim1: int,
        dim2: int,
    ) -> torch.Tensor:
        """
        Pure PyTorch voxelization for 3D Gaussians.
        
        CT Formula: O[i,j,k] = Σ_g (opacity_g × exp(-sigma_g))
        where sigma_g = 0.5 × delta^T × Σ^{-1} × delta
        
        Args:
            means3d: (N, 3) Gaussian centers [x, y, z]
            conics: (N, 6) Inverse covariance [a, p, q, b, r, c]
            opacities: (N,) Gaussian opacities
            dim0: Volume depth (z)
            dim1: Volume height (y)
            dim2: Volume width (x)
            
        Returns:
            render_output: (dim0, dim1, dim2) Rendered volume
        """
        N = means3d.shape[0]
        device = means3d.device
        dtype = means3d.dtype
        
        # Create voxel coordinate grid with 0.5 offset for voxel centers
        z_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        y_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim2, device=device, dtype=dtype) + 0.5
        
        # meshgrid: output order is (z, y, x) for indexing='ij'
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
        # Stack to get (M, 3) with [x, y, z] ordering where M = dim0*dim1*dim2
        voxel_coords = torch.stack([
            xx.reshape(-1),
            yy.reshape(-1),
            zz.reshape(-1)
        ], dim=-1)  # (M, 3)
        
        M = voxel_coords.shape[0]
        
        # Compute delta = voxel - mean: (M, N, 3)
        delta = voxel_coords.unsqueeze(1) - means3d.unsqueeze(0)  # (M, N, 3)
        dx = delta[..., 0]  # (M, N)
        dy = delta[..., 1]  # (M, N)
        dz = delta[..., 2]  # (M, N)
        
        # Extract conic elements: (N, 6) -> broadcast to (1, N)
        a = conics[:, 0].unsqueeze(0)
        p = conics[:, 1].unsqueeze(0)
        q = conics[:, 2].unsqueeze(0)
        b = conics[:, 3].unsqueeze(0)
        r = conics[:, 4].unsqueeze(0)
        c = conics[:, 5].unsqueeze(0)
        
        # sigma = 0.5*(a*dx² + b*dy² + c*dz²) + p*dx*dy + q*dx*dz + r*dy*dz
        sigma = 0.5 * (a * dx * dx + b * dy * dy + c * dz * dz) + \
                p * dx * dy + q * dx * dz + r * dy * dz
        
        # G = opacity * exp(-sigma)
        G = opacities.unsqueeze(0) * torch.exp(-sigma)  # (M, N)
        
        # Apply numerical stability mask
        mask = (sigma >= 0) & (G.abs() >= 1e-5)
        G = torch.where(mask, G, torch.zeros_like(G))
        
        # Sum over Gaussians and reshape
        output = G.sum(dim=1).reshape(dim0, dim1, dim2)
        
        return output
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only."""
        # Extract parameters
        if isinstance(params, GS3DSmoothingParams):
            means3d = params.means3d
            conics = params.conics
            opacities = params.opacities
        else:
            means3d = params.get('means3d').data
            conics = params.get('conics').data
            opacities = params.get('opacities').data
        
        # Extract data
        if isinstance(data, GS3DSmoothingData):
            dim0 = data.dim0
            dim1 = data.dim1
            dim2 = data.dim2
        else:
            dim0 = data.get('dim0')
            dim1 = data.get('dim1')
            dim2 = data.get('dim2')
        
        # Get config
        tv_weight = config.get('tv_weight', self.tv_weight) if config else self.tv_weight
        
        # Voxelize
        render_output = self._voxelize_torch(means3d, conics, opacities, dim0, dim1, dim2)
        
        # Compute TV loss
        tv_loss = compute_tv_loss(render_output)
        loss = tv_weight * tv_loss
        
        return ForwardResult(
            outputs={
                'loss': loss,
                'render_output': render_output.detach(),
                'tv_loss': tv_loss.detach(),
            },
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward."""
        # Extract parameters with gradients
        if isinstance(params, GS3DSmoothingParams):
            means3d = params.means3d.clone().requires_grad_(True)
            conics = params.conics.clone().requires_grad_(True)
            opacities = params.opacities.clone().requires_grad_(True)
        else:
            means3d = params.get('means3d').data.clone().requires_grad_(True)
            conics = params.get('conics').data.clone().requires_grad_(True)
            opacities = params.get('opacities').data.clone().requires_grad_(True)
        
        # Extract data
        if isinstance(data, GS3DSmoothingData):
            dim0 = data.dim0
            dim1 = data.dim1
            dim2 = data.dim2
        else:
            dim0 = data.get('dim0')
            dim1 = data.get('dim1')
            dim2 = data.get('dim2')
        
        # Get config
        tv_weight = config.get('tv_weight', self.tv_weight) if config else self.tv_weight
        
        # Forward pass
        render_output = self._voxelize_torch(means3d, conics, opacities, dim0, dim1, dim2)
        
        # Compute loss
        tv_loss = compute_tv_loss(render_output)
        loss = tv_weight * tv_loss
        
        # Backward pass
        loss.backward()
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss.detach(),
                'render_output': render_output.detach(),
                'tv_loss': tv_loss.detach(),
            },
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('means3d', means3d.grad.detach())
        grad_result.set('conics', conics.grad.detach())
        grad_result.set('opacities', opacities.grad.detach())
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward using torch autograd.
        
        Simplified Approach:
        For each Gaussian, concatenate all parameters into a single vector θ = [means3d, conics, opacities]
        (10 elements total: 3 + 6 + 1), compute the full 10×10 Hessian, then extract blocks.
        
        This matches CUDA output structure:
        - means3d: (N, 6) upper triangle [H00, H01, H02, H11, H12, H22]
        - conics: (N, 21) upper triangle of 6×6 matrix
        - opacities: (N,) scalar per Gaussian
        - Cross-terms: op_mean (N, 3), op_conic (N, 6), mean_conic (N, 18)
        
        Parameter layout in θ:
          [0:3]  = means3d (x, y, z)
          [3:9]  = conics (a, p, q, b, r, c) = (c00, c01, c02, c11, c12, c22)
          [9]    = opacity
        
        Hessian block extraction:
          H[0:3, 0:3]   -> h_means3d (upper triangle: 6 elements)
          H[3:9, 3:9]   -> h_conics (upper triangle: 21 elements)
          H[9, 9]       -> h_opacities (1 element)
          H[9, 0:3]     -> h_op_mean (3 elements)
          H[9, 3:9]     -> h_op_conic (6 elements)
          H[0:3, 3:9]   -> h_mean_conic (18 elements, row-major)
        """
        # Extract parameters with gradients
        if isinstance(params, GS3DSmoothingParams):
            means3d = params.means3d.clone().requires_grad_(True)
            conics = params.conics.clone().requires_grad_(True)
            opacities = params.opacities.clone().requires_grad_(True)
        else:
            means3d = params.get('means3d').data.clone().requires_grad_(True)
            conics = params.get('conics').data.clone().requires_grad_(True)
            opacities = params.get('opacities').data.clone().requires_grad_(True)
        
        # Extract data
        if isinstance(data, GS3DSmoothingData):
            dim0 = data.dim0
            dim1 = data.dim1
            dim2 = data.dim2
        else:
            dim0 = data.get('dim0')
            dim1 = data.get('dim1')
            dim2 = data.get('dim2')
        
        # Get config
        tv_weight = config.get('tv_weight', self.tv_weight) if config else self.tv_weight
        
        N = means3d.shape[0]
        device = means3d.device
        dtype = means3d.dtype
        
        # Forward pass
        render_output = self._voxelize_torch(means3d, conics, opacities, dim0, dim1, dim2)
        
        # Compute loss using smooth L1 for non-zero Hessian (matches CUDA bwd2)
        tv_loss = compute_tv_loss(render_output, smooth=True, eps=1e-6)
        loss = tv_weight * tv_loss
        
        # First-order gradients (with create_graph=True for second-order)
        v_means3d, v_conics, v_opacities = torch.autograd.grad(
            loss, [means3d, conics, opacities],
            create_graph=True
        )
        
        # Helper: Extract upper triangle indices from matrix
        def upper_triangle_indices(n: int) -> list:
            """Return list of (i, j) pairs for upper triangle including diagonal."""
            indices = []
            for i in range(n):
                for j in range(i, n):
                    indices.append((i, j))
            return indices
        
        # Precompute upper triangle indices
        mean_utri = upper_triangle_indices(3)   # 6 elements
        conic_utri = upper_triangle_indices(6)  # 21 elements
        
        # Allocate Hessian output tensors
        h_means3d = torch.zeros(N, 6, device=device, dtype=dtype)      # upper triangle of 3x3
        h_conics = torch.zeros(N, 21, device=device, dtype=dtype)      # upper triangle of 6x6
        h_opacities = torch.zeros(N, device=device, dtype=dtype)       # scalar
        h_op_mean = torch.zeros(N, 3, device=device, dtype=dtype)      # cross-term
        h_op_conic = torch.zeros(N, 6, device=device, dtype=dtype)     # cross-term
        h_mean_conic = torch.zeros(N, 18, device=device, dtype=dtype)  # cross-term (3x6)
        
        # Process each Gaussian
        for n in range(N):
            # Concatenate gradients for this Gaussian: [grad_mean, grad_conic, grad_opacity]
            grad_vec = torch.cat([
                v_means3d[n],          # (3,)
                v_conics[n],           # (6,)
                v_opacities[n:n+1],    # (1,)
            ])  # (10,)
            
            # Build 10×10 Hessian matrix by computing gradient of each gradient component
            H = torch.zeros(10, 10, device=device, dtype=dtype)
            
            for i in range(10):
                # Compute d(grad_vec[i])/d(theta) where theta = [means3d[n], conics[n], opacities[n]]
                grads = torch.autograd.grad(
                    grad_vec[i],
                    [means3d, conics, opacities],
                    retain_graph=True,
                    allow_unused=True
                )
                
                # Extract relevant row of Hessian for Gaussian n
                if grads[0] is not None:
                    H[i, 0:3] = grads[0][n]
                if grads[1] is not None:
                    H[i, 3:9] = grads[1][n]
                if grads[2] is not None:
                    H[i, 9] = grads[2][n]
            
            # Extract blocks from the 10×10 Hessian
            
            # means3d Hessian: H[0:3, 0:3] upper triangle -> 6 elements
            for idx, (i, j) in enumerate(mean_utri):
                h_means3d[n, idx] = H[i, j]
            
            # conics Hessian: H[3:9, 3:9] upper triangle -> 21 elements
            for idx, (i, j) in enumerate(conic_utri):
                h_conics[n, idx] = H[3 + i, 3 + j]
            
            # opacities Hessian: H[9, 9] -> scalar
            h_opacities[n] = H[9, 9]
            
            # Cross-term opacity × means3d: H[9, 0:3] -> 3 elements
            h_op_mean[n] = H[9, 0:3]
            
            # Cross-term opacity × conics: H[9, 3:9] -> 6 elements
            h_op_conic[n] = H[9, 3:9]
            
            # Cross-term means3d × conics: H[0:3, 3:9] -> 18 elements (row-major)
            h_mean_conic[n] = H[0:3, 3:9].flatten()
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss.detach(),
                'render_output': render_output.detach(),
                'tv_loss': tv_loss.detach(),
            },
            intermediates={}
        )
        
        grad_result = GradResult()
        grad_result.set('means3d', v_means3d.detach())
        grad_result.set('conics', v_conics.detach())
        grad_result.set('opacities', v_opacities.detach())
        
        # Extract diagonal Hessians for optimizer (matching gradient shapes)
        # h_means3d: (N, 6) [H00, H01, H02, H11, H12, H22] -> (N, 3) [H00, H11, H22]
        h_means3d_diag = torch.stack([h_means3d[:, 0], h_means3d[:, 3], h_means3d[:, 5]], dim=-1)  # (N, 3)
        
        # h_conics: (N, 21) upper triangle of 6x6 -> (N, 6) diagonal
        # Diagonal at: (0,0)=0, (1,1)=6, (2,2)=11, (3,3)=15, (4,4)=18, (5,5)=20
        h_conics_diag = torch.stack([
            h_conics[:, 0],   # (0,0)
            h_conics[:, 6],   # (1,1)
            h_conics[:, 11],  # (2,2)
            h_conics[:, 15],  # (3,3)
            h_conics[:, 18],  # (4,4)
            h_conics[:, 20],  # (5,5)
        ], dim=-1)  # (N, 6)
        
        hess_result = HessResult()
        # Return full upper triangular Hessians
        hess_result.set('means3d', h_means3d)      # (N, 6) upper triangle
        hess_result.set('conics', h_conics)        # (N, 21) upper triangle
        hess_result.set('opacities', h_opacities)
        hess_result.set('op_mean', h_op_mean)
        hess_result.set('op_conic', h_op_conic)
        hess_result.set('mean_conic', h_mean_conic)
        # Also provide diagonal versions for convenience (used by optimizer)
        hess_result.set('means3d_diag', h_means3d_diag)
        hess_result.set('conics_diag', h_conics_diag)
        
        return forward_result, grad_result, hess_result


# ============================================================================
# CUDA Implementation
# ============================================================================

class GS3DSmoothingCUDA(SecondForwardBackwardBase):
    """
    3D Gaussian Splatting with TV Smoothing using CUDA kernels.
    
    This implementation uses custom CUDA kernels for:
    - Forward voxelization (voxelize_to_voxels_fwd)
    - First-order backward (voxelize_to_voxels_bwd + volume_loss_bwd)
    - Second-order backward (voxelize_to_voxels_bwd2 + volume_loss_bwd2)
    
    The pipeline chains:
    1. Voxelization: Gaussian params -> Rendered volume
    2. TV loss: Volume -> Scalar loss
    
    For backward passes, we use chain rule:
    - v_params = v_volume @ (∂volume/∂params)
    - h_params = (∂volume/∂params)^T @ h_volume @ (∂volume/∂params) + ...
    """
    
    def __init__(
        self,
        tv_weight: float = 1.0,
        tile_size: int = 4,
        eps: float = 1e-6,
    ):
        self.tv_weight = tv_weight
        self.tile_size = tile_size
        self.eps = eps
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only."""
        from ..cuda._wrapper import (
            isect_tiles_3d,
            isect_offset_encode_3d,
            voxelize_to_voxels_fwd,
            volume_loss_fwd,
        )
        
        # Extract parameters
        if isinstance(params, GS3DSmoothingParams):
            means3d = params.means3d
            conics = params.conics
            opacities = params.opacities
        else:
            means3d = params.get('means3d').data
            conics = params.get('conics').data
            opacities = params.get('opacities').data
        
        # Extract data
        if isinstance(data, GS3DSmoothingData):
            radii = data.radii
            dim0 = data.dim0
            dim1 = data.dim1
            dim2 = data.dim2
            tile_size = data.tile_size
        else:
            radii = data.get('radii')
            dim0 = data.get('dim0')
            dim1 = data.get('dim1')
            dim2 = data.get('dim2')
            tile_size = data.get('tile_size', self.tile_size)
        
        # Get config
        tv_weight = config.get('tv_weight', self.tv_weight) if config else self.tv_weight
        
        n_tiles_dim2 = (dim2 + tile_size - 1) // tile_size
        n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
        n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
        
        # Compute tile intersections
        _, isect_ids, flatten_ids = isect_tiles_3d(
            means3d, radii, tile_size, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        tile_offsets = isect_offset_encode_3d(
            isect_ids, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        
        # Voxelization
        render_output = voxelize_to_voxels_fwd(
            means3d, conics, opacities,
            tile_offsets, flatten_ids,
            dim2, dim1, dim0, tile_size
        )
        
        # TV loss (add batch dimension for CUDA API)
        render_4d = render_output.unsqueeze(0)  # (1, dim0, dim1, dim2)
        eps = config.get('eps', self.eps) if config else self.eps
        loss, tv_loss, tv_map = volume_loss_fwd(render_4d, eps=eps, lambda_tv=tv_weight)
        
        return ForwardResult(
            outputs={
                'loss': loss,
                'render_output': render_output.detach(),
                'tv_loss': tv_loss.detach(),
            },
            intermediates={
                'tile_offsets': tile_offsets,
                'flatten_ids': flatten_ids,
            }
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward."""
        from ..cuda._wrapper import (
            isect_tiles_3d,
            isect_offset_encode_3d,
            voxelize_to_voxels_fwd,
            voxelize_to_voxels_bwd,
            volume_loss_fwd,
            volume_loss_bwd,
        )
        
        # Extract parameters
        if isinstance(params, GS3DSmoothingParams):
            means3d = params.means3d
            conics = params.conics
            opacities = params.opacities
        else:
            means3d = params.get('means3d').data
            conics = params.get('conics').data
            opacities = params.get('opacities').data
        
        # Extract data
        if isinstance(data, GS3DSmoothingData):
            radii = data.radii
            dim0 = data.dim0
            dim1 = data.dim1
            dim2 = data.dim2
            tile_size = data.tile_size
        else:
            radii = data.get('radii')
            dim0 = data.get('dim0')
            dim1 = data.get('dim1')
            dim2 = data.get('dim2')
            tile_size = data.get('tile_size', self.tile_size)
        
        # Get config
        tv_weight = config.get('tv_weight', self.tv_weight) if config else self.tv_weight
        
        n_tiles_dim2 = (dim2 + tile_size - 1) // tile_size
        n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
        n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
        
        # Compute tile intersections
        _, isect_ids, flatten_ids = isect_tiles_3d(
            means3d, radii, tile_size, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        tile_offsets = isect_offset_encode_3d(
            isect_ids, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        
        # Forward: Voxelization
        render_output = voxelize_to_voxels_fwd(
            means3d, conics, opacities,
            tile_offsets, flatten_ids,
            dim2, dim1, dim0, tile_size
        )
        
        # Forward: TV loss
        render_4d = render_output.unsqueeze(0)
        eps = config.get('eps', self.eps) if config else self.eps
        loss, tv_loss, tv_map = volume_loss_fwd(render_4d, eps=eps, lambda_tv=tv_weight)
        
        # Backward: TV loss -> v_render_output
        grad_output = torch.ones(1, device=render_output.device, dtype=render_output.dtype)
        v_render_4d = volume_loss_bwd(render_4d, grad_output, eps=eps, lambda_tv=tv_weight)
        v_render_output = v_render_4d.squeeze(0)  # (dim0, dim1, dim2)
        
        # Backward: Voxelization -> v_params
        v_means3d, v_conics, v_opacities = voxelize_to_voxels_bwd(
            means3d, conics, opacities,
            tile_offsets, flatten_ids, v_render_output,
            dim2, dim1, dim0, tile_size
        )
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss,
                'render_output': render_output.detach(),
                'tv_loss': tv_loss.detach(),
            },
            intermediates={
                'tile_offsets': tile_offsets,
                'flatten_ids': flatten_ids,
                'v_render_output': v_render_output,
            }
        )
        
        grad_result = GradResult()
        grad_result.set('means3d', v_means3d)
        grad_result.set('conics', v_conics)
        grad_result.set('opacities', v_opacities)
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + first-order + second-order backward."""
        from ..cuda._wrapper import (
            isect_tiles_3d,
            isect_offset_encode_3d,
            voxelize_to_voxels_fwd,
            voxelize_to_voxels_bwd2,
            volume_loss_fwd,
            volume_loss_bwd,
            volume_loss_bwd2,
        )
        
        # Extract parameters
        if isinstance(params, GS3DSmoothingParams):
            means3d = params.means3d
            conics = params.conics
            opacities = params.opacities
        else:
            means3d = params.get('means3d').data
            conics = params.get('conics').data
            opacities = params.get('opacities').data
        
        # Extract data
        if isinstance(data, GS3DSmoothingData):
            radii = data.radii
            dim0 = data.dim0
            dim1 = data.dim1
            dim2 = data.dim2
            tile_size = data.tile_size
        else:
            radii = data.get('radii')
            dim0 = data.get('dim0')
            dim1 = data.get('dim1')
            dim2 = data.get('dim2')
            tile_size = data.get('tile_size', self.tile_size)
        
        # Get config
        tv_weight = config.get('tv_weight', self.tv_weight) if config else self.tv_weight
        
        n_tiles_dim2 = (dim2 + tile_size - 1) // tile_size
        n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
        n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
        
        # Compute tile intersections
        _, isect_ids, flatten_ids = isect_tiles_3d(
            means3d, radii, tile_size, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        tile_offsets = isect_offset_encode_3d(
            isect_ids, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        
        # Forward: Voxelization
        render_output = voxelize_to_voxels_fwd(
            means3d, conics, opacities,
            tile_offsets, flatten_ids,
            dim2, dim1, dim0, tile_size
        )
        
        # Forward: TV loss
        render_4d = render_output.unsqueeze(0)
        eps = config.get('eps', self.eps) if config else self.eps
        loss, tv_loss, tv_map = volume_loss_fwd(render_4d, eps=eps, lambda_tv=tv_weight)
        
        # Backward: TV loss gradients
        grad_output = torch.ones(1, device=render_output.device, dtype=render_output.dtype)
        v_render_4d = volume_loss_bwd(render_4d, grad_output, eps=eps, lambda_tv=tv_weight)
        v_render_output = v_render_4d.squeeze(0)
        
        # Backward: TV loss Hessian diagonal
        h_render_4d = volume_loss_bwd2(render_4d, grad_output, eps=eps, lambda_tv=tv_weight)
        h_render_output = h_render_4d.squeeze(0)
        
        # Second-order backward: Voxelization
        (v_means3d, v_conics, v_opacities,
         h_means3d, h_conics, h_opacities,
         h_op_mean, h_op_conic, h_mean_conic) = voxelize_to_voxels_bwd2(
            means3d, conics, opacities,
            tile_offsets, flatten_ids,
            v_render_output, h_render_output,
            dim2, dim1, dim0, tile_size
        )
        
        forward_result = ForwardResult(
            outputs={
                'loss': loss,
                'render_output': render_output.detach(),
                'tv_loss': tv_loss.detach(),
            },
            intermediates={
                'tile_offsets': tile_offsets,
                'flatten_ids': flatten_ids,
            }
        )
        
        grad_result = GradResult()
        grad_result.set('means3d', v_means3d)
        grad_result.set('conics', v_conics)
        grad_result.set('opacities', v_opacities)
        
        # Extract diagonal Hessians for optimizer (matching gradient shapes)
        # h_means3d: (N, 6) [H00, H01, H02, H11, H12, H22] -> (N, 3) [H00, H11, H22]
        # h_conics: (N, 21) upper triangle of 6x6 -> (N, 6) diagonal
        # h_opacities: (N,) already scalar per Gaussian
        
        # means3d diagonal: indices [0, 3, 5] from upper triangle
        h_means3d_diag = torch.stack([h_means3d[:, 0], h_means3d[:, 3], h_means3d[:, 5]], dim=-1)  # (N, 3)
        
        # conics diagonal: indices [0, 6, 11, 15, 18, 20] from 21-element upper triangle
        # Upper triangle of 6x6 symmetric matrix stored as:
        # Row 0: 0,1,2,3,4,5 (6 elements)
        # Row 1: 6,7,8,9,10 (5 elements)
        # Row 2: 11,12,13,14 (4 elements)
        # Row 3: 15,16,17 (3 elements)
        # Row 4: 18,19 (2 elements)
        # Row 5: 20 (1 element)
        # Diagonal at: (0,0)=0, (1,1)=6, (2,2)=11, (3,3)=15, (4,4)=18, (5,5)=20
        h_conics_diag = torch.stack([
            h_conics[:, 0],   # (0,0)
            h_conics[:, 6],   # (1,1)
            h_conics[:, 11],  # (2,2)
            h_conics[:, 15],  # (3,3)
            h_conics[:, 18],  # (4,4)
            h_conics[:, 20],  # (5,5)
        ], dim=-1)  # (N, 6)
        
        hess_result = HessResult()
        # Return full upper triangular Hessians (matching Autograd output)
        # Optimizer will extract diagonals as needed
        hess_result.set('means3d', h_means3d)  # (N, 6) upper triangle
        hess_result.set('conics', h_conics)    # (N, 21) upper triangle
        hess_result.set('opacities', h_opacities)
        hess_result.set('op_mean', h_op_mean)
        hess_result.set('op_conic', h_op_conic)
        hess_result.set('mean_conic', h_mean_conic)
        # Also provide diagonal versions for convenience (used by optimizer)
        hess_result.set('means3d_diag', h_means3d_diag)
        hess_result.set('conics_diag', h_conics_diag)
        
        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Functions
# ============================================================================

def create_gs3d_smoothing_autograd(
    tv_weight: float = 1.0,
    tile_size: int = 4,
) -> GS3DSmoothingAutograd:
    """Create 3D Gaussian Splatting with Smoothing operator (autograd version)."""
    return GS3DSmoothingAutograd(
        tv_weight=tv_weight,
        tile_size=tile_size,
    )


def create_gs3d_smoothing_cuda(
    tv_weight: float = 1.0,
    tile_size: int = 4,
    eps: float = 1e-6,
) -> GS3DSmoothingCUDA:
    """Create 3D Gaussian Splatting with Smoothing operator (CUDA version)."""
    return GS3DSmoothingCUDA(
        tv_weight=tv_weight,
        tile_size=tile_size,
        eps=eps,
    )


__all__ = [
    # Data structures
    'GS3DSmoothingParams',
    'GS3DSmoothingData',
    'GS3DSmoothingConfig',
    # Operators
    'GS3DSmoothingAutograd',
    'GS3DSmoothingCUDA',
    # Factory functions
    'create_gs3d_smoothing_autograd',
    'create_gs3d_smoothing_cuda',
]
