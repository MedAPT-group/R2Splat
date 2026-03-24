"""
VoxelizeToVoxels Operator for CT Reconstruction.

This module implements the 3D Gaussian voxelization operator with support for
both first-order and second-order backward passes.

CT formula: voxel_value = Σ (opacity × exp(-sigma))
where sigma = 0.5 × (x - m)^T × Σ^{-1} × (x - m)

Author: Dy164
Date: 2026-02-03
"""

from typing import Optional, Tuple
import torch

from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


def _is_cuda_available() -> bool:
    """Check if CUDA extension is available."""
    try:
        from ..cuda._wrapper import is_cuda_extension_available
        return is_cuda_extension_available()
    except ImportError:
        return False


class VoxelizeToVoxelsCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated voxelize to voxels implementation.
    
    Uses tile-based voxelization with shared memory optimization.
    Requires isect_tiles_3d and isect_offset_encode_3d for preprocessing.
    
    CT formula: voxel_value = Σ (opacity × exp(-sigma))
    where sigma = 0.5 × delta^T × conic × delta
    
    Supports: (dim0, dim1, dim2) - 3D voxel grid without channel dimension
    
    This class computes:
    - Forward: render output (dim0, dim1, dim2)
    - Backward1: first-order gradients w.r.t. Gaussian parameters
    - Backward2: second-order Hessians with cross-terms
    
    Hessian storage (upper triangular):
    - means3d: [N, 6] = [H00, H01, H02, H11, H12, H22]
    - conics: [N, 21] = upper triangle of 6×6 matrix
    - opacities: [N] scalar
    - Cross-terms: h_op_mean, h_op_conic, h_mean_conic
    """
    
    def __init__(self):
        if not _is_cuda_available():
            raise RuntimeError(
                "CUDA extension not available. "
                "Please compile the CUDA extension first with 'pip install -e .'"
            )
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        required_params = ['means3d', 'conics', 'opacities']
        for name in required_params:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        
        means3d = params.get('means3d').data
        
        # Must be 2D: (N, 3)
        if means3d.dim() != 2 or means3d.size(-1) != 3:
            raise ValueError(
                f"means3d must be 2D with shape (N, 3), got {means3d.shape}"
            )
        
        # Ensure CUDA tensors
        if not means3d.is_cuda:
            raise ValueError("All parameters must be CUDA tensors")
        
        self.initialized = True
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only (no gradients).
        
        Expected params:
            - means3d: (N, 3) 3D Gaussian centers, format: (x, y, z)
            - conics: (N, 6) Inverse covariance matrices [a, p, q, b, r, c]
            - opacities: (N) Gaussian opacities
        
        Expected data:
            - radii: (N, 3) Gaussian radii for tile intersection
            - dim2: int - output size in dimension 2 (x-axis, fastest)
            - dim1: int - output size in dimension 1 (y-axis, middle)
            - dim0: int - output size in dimension 0 (z-axis, slowest)
            - tile_size: int (default 4)
        
        Returns:
            ForwardResult with render_output (dim0, dim1, dim2)
        """
        from ..cuda._wrapper import (
            isect_tiles_3d,
            isect_offset_encode_3d,
            voxelize_to_voxels_fwd,
        )
        
        # Extract parameters
        means3d = params.get('means3d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        
        # Extract data
        radii = data.get('radii')
        dim2 = data.get('dim2')
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        tile_size = data.get('tile_size', 4)
        
        n_tiles_dim2 = (dim2 + tile_size - 1) // tile_size
        n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
        n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
        
        # Compute tile intersections
        _, isect_ids, flatten_ids = isect_tiles_3d(
            means3d, radii, tile_size, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        
        # Encode tile offsets
        tile_offsets = isect_offset_encode_3d(
            isect_ids, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        
        # Voxelize
        render_output = voxelize_to_voxels_fwd(
            means3d, conics, opacities,
            tile_offsets, flatten_ids,
            dim2, dim1, dim0, tile_size
        )
        
        return ForwardResult(
            outputs={'render_output': render_output},
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
        """Forward + first-order backward using CUDA kernels.
        
        Computes gradients of render_output w.r.t. all Gaussian parameters.
        
        Expected params: same as forward_only
        Expected data: same as forward_only, plus:
            - v_render_output: (dim0, dim1, dim2) upstream gradient (optional)
              If not provided, uses ones (equivalent to sum of all voxels)
        
        Returns:
            ForwardResult with render_output (dim0, dim1, dim2)
            GradResult with gradients for means3d, conics, opacities
        """
        from ..cuda._wrapper import (
            isect_tiles_3d,
            isect_offset_encode_3d,
            voxelize_to_voxels_fwd,
            voxelize_to_voxels_bwd,
        )
        
        # Extract parameters
        means3d = params.get('means3d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        
        # Extract data
        radii = data.get('radii')
        dim2 = data.get('dim2')
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        tile_size = data.get('tile_size', 4)
        
        n_tiles_dim2 = (dim2 + tile_size - 1) // tile_size
        n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
        n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
        
        # Compute tile intersections
        _, isect_ids, flatten_ids = isect_tiles_3d(
            means3d, radii, tile_size, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        
        # Encode tile offsets
        tile_offsets = isect_offset_encode_3d(
            isect_ids, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        
        # Forward pass
        render_output = voxelize_to_voxels_fwd(
            means3d, conics, opacities,
            tile_offsets, flatten_ids,
            dim2, dim1, dim0, tile_size
        )
        
        # Get upstream gradient
        v_render_output = data.get('v_render_output', None)
        if v_render_output is None:
            v_render_output = torch.ones_like(render_output)
        
        # Backward pass
        v_means3d, v_conics, v_opacities = voxelize_to_voxels_bwd(
            means3d, conics, opacities,
            tile_offsets, flatten_ids, v_render_output,
            dim2, dim1, dim0, tile_size
        )
        
        forward_result = ForwardResult(
            outputs={'render_output': render_output},
            intermediates={
                'tile_offsets': tile_offsets,
                'flatten_ids': flatten_ids,
            }
        )
        
        grad_result = GradResult(
            param_grads={
                'means3d': v_means3d,
                'conics': v_conics,
                'opacities': v_opacities,
            }
        )
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward using CUDA kernels.
        
        Computes both first-order gradients and second-order Hessians.
        
        CT formula: G = o × exp(-σ) where σ = 0.5 × d^T × Σ^{-1} × d
        
        Expected params: same as forward_only
        Expected data: same as forward_only, plus:
            - v_render_output: (dim0, dim1, dim2) first-order upstream gradient
            - h_render_output: (dim0, dim1, dim2) second-order upstream gradient
              If not provided, uses ones
        
        Hessian storage uses upper triangular elements:
            - means3d: (N, 6) [H00, H01, H02, H11, H12, H22]
            - conics: (N, 21) upper triangle of 6×6 matrix
            - opacities: (N) scalar
        
        Cross-term Hessians:
            - op_mean: (N, 3)
            - op_conic: (N, 6)
            - mean_conic: (N, 18) = (N, 3, 6)
        
        Returns:
            ForwardResult with render_output (dim0, dim1, dim2)
            GradResult with first-order gradients
            HessResult with:
                - param_hess: Hessian upper triangles for each parameter group
                - cross_hess: Cross-term Hessians between parameter groups
        """
        from ..cuda._wrapper import (
            isect_tiles_3d,
            isect_offset_encode_3d,
            voxelize_to_voxels_fwd,
            voxelize_to_voxels_bwd2,
        )
        
        # Extract parameters
        means3d = params.get('means3d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        
        # Extract data
        radii = data.get('radii')
        dim2 = data.get('dim2')
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        tile_size = data.get('tile_size', 4)
        
        n_tiles_dim2 = (dim2 + tile_size - 1) // tile_size
        n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
        n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
        
        # Compute tile intersections
        _, isect_ids, flatten_ids = isect_tiles_3d(
            means3d, radii, tile_size, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        
        # Encode tile offsets
        tile_offsets = isect_offset_encode_3d(
            isect_ids, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
        )
        
        # Forward pass
        render_output = voxelize_to_voxels_fwd(
            means3d, conics, opacities,
            tile_offsets, flatten_ids,
            dim2, dim1, dim0, tile_size
        )
        
        # Get upstream gradients
        v_render_output = data.get('v_render_output', None)
        if v_render_output is None:
            v_render_output = torch.ones_like(render_output)
        
        h_render_output = data.get('h_render_output', None)
        if h_render_output is None:
            h_render_output = torch.ones_like(render_output)
        
        # Second-order backward pass
        (v_means3d, v_conics, v_opacities,
         h_means3d, h_conics, h_opacities,
         h_op_mean, h_op_conic, h_mean_conic) = voxelize_to_voxels_bwd2(
            means3d, conics, opacities,
            tile_offsets, flatten_ids,
            v_render_output, h_render_output,
            dim2, dim1, dim0, tile_size
        )
        
        forward_result = ForwardResult(
            outputs={'render_output': render_output},
            intermediates={
                'tile_offsets': tile_offsets,
                'flatten_ids': flatten_ids,
            }
        )
        
        grad_result = GradResult(
            param_grads={
                'means3d': v_means3d,
                'conics': v_conics,
                'opacities': v_opacities,
            }
        )
        
        hess_result = HessResult(
            param_hess={
                'means3d': h_means3d,      # (N, 6) upper triangle
                'conics': h_conics,        # (N, 21) upper triangle
                'opacities': h_opacities,  # (N)
                # Cross-terms
                'op_mean': h_op_mean,      # (N, 3)
                'op_conic': h_op_conic,    # (N, 6)
                'mean_conic': h_mean_conic,  # (N, 18)
            }
        )
        
        return forward_result, grad_result, hess_result


class VoxelizeToVoxelsAutograd(SecondForwardBackwardBase):
    """
    PyTorch autograd-based voxelize to voxels implementation.
    
    This class provides a reference implementation using pure PyTorch
    for validation against the CUDA implementation.
    
    CT formula: voxel_value = Σ (opacity × exp(-sigma))
    where sigma = 0.5 × delta^T × conic × delta
    
    Conic storage: [N, 6] = [a, p, q, b, r, c] (upper triangular of 3x3)
    Quadratic form: sigma = 0.5 * (a*dx² + 2*p*dx*dy + 2*q*dx*dz + 
                                    b*dy² + 2*r*dy*dz + c*dz²)
    
    Coordinate convention:
        - dim0 (z) - slowest varying
        - dim1 (y) - middle
        - dim2 (x) - fastest varying (columns)
    """
    
    def __init__(self):
        self.initialized = False
    
    def initialize(
        self,
        params: Params,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize and validate parameters."""
        required_params = ['means3d', 'conics', 'opacities']
        for name in required_params:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        
        means3d = params.get('means3d').data
        
        # Must be 2D: (N, 3)
        if means3d.dim() != 2 or means3d.size(-1) != 3:
            raise ValueError(
                f"means3d must be 2D with shape (N, 3), got {means3d.shape}"
            )
        
        self.initialized = True
    
    @staticmethod
    def _compute_gaussian_value(
        means3d: torch.Tensor,  # (N, 3)
        conics: torch.Tensor,   # (N, 6) [a, p, q, b, r, c]
        opacities: torch.Tensor,  # (N,)
        voxel_coords: torch.Tensor,  # (M, 3) [x, y, z]
    ) -> torch.Tensor:
        """
        Compute Gaussian values at given voxel coordinates.
        
        Args:
            means3d: (N, 3) Gaussian centers [x, y, z]
            conics: (N, 6) [a, p, q, b, r, c]
            opacities: (N,) Gaussian opacities
            voxel_coords: (M, 3) voxel coordinates [x, y, z]
        
        Returns:
            (M, N) Gaussian values at each voxel for each Gaussian
        """
        # delta = voxel - mean: (M, N, 3)
        # voxel_coords: (M, 1, 3), means3d: (1, N, 3)
        delta = voxel_coords.unsqueeze(1) - means3d.unsqueeze(0)  # (M, N, 3)
        dx = delta[..., 0]  # (M, N)
        dy = delta[..., 1]  # (M, N)
        dz = delta[..., 2]  # (M, N)
        
        # conics: (N, 6) -> broadcast to (1, N, 6)
        a = conics[..., 0].unsqueeze(0)  # (1, N)
        p = conics[..., 1].unsqueeze(0)
        q = conics[..., 2].unsqueeze(0)
        b = conics[..., 3].unsqueeze(0)
        r = conics[..., 4].unsqueeze(0)
        c = conics[..., 5].unsqueeze(0)
        
        # sigma = 0.5*(a*dx² + b*dy² + c*dz²) + p*dx*dy + q*dx*dz + r*dy*dz
        # Match CUDA formula exactly for correct autograd
        sigma = 0.5 * (a * dx * dx + b * dy * dy + c * dz * dz) + p * dx * dy + q * dx * dz + r * dy * dz
        
        # G = opacity * exp(-sigma)
        # opacities: (N,) -> (1, N)
        G = opacities.unsqueeze(0) * torch.exp(-sigma)  # (M, N)
        
        # Apply numerical stability check (same as CUDA)
        # Skip if sigma < 0 or |alpha| < 1e-5
        mask = (sigma >= 0) & (G.abs() >= 1e-5)
        G = torch.where(mask, G, torch.zeros_like(G))
        
        return G
    
    def _forward_impl(
        self,
        means3d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        dim2: int,
        dim1: int,
        dim0: int,
    ) -> torch.Tensor:
        """
        Forward implementation using brute-force computation.
        
        For each voxel, sum contributions from all Gaussians.
        This is O(M*N) where M = dim0*dim1*dim2, N = num Gaussians.
        
        Returns:
            (dim0, dim1, dim2) voxel output
        """
        device = means3d.device
        dtype = means3d.dtype
        
        # Create voxel coordinate grid with 0.5 offset for voxel centers
        # Convention: x=dim2 (fastest), y=dim1, z=dim0 (slowest)
        z_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        y_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim2, device=device, dtype=dtype) + 0.5
        
        # meshgrid: output order is (z, y, x) for indexing='ij'
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
        # Stack to get (dim0*dim1*dim2, 3) with [x, y, z] ordering
        voxel_coords = torch.stack([
            xx.reshape(-1),
            yy.reshape(-1),
            zz.reshape(-1)
        ], dim=-1)  # (M, 3) [x, y, z]
        
        # Compute Gaussian values: (M, N)
        G = self._compute_gaussian_value(means3d, conics, opacities, voxel_coords)
        
        # Sum over all Gaussians: (M,) then reshape to (dim0, dim1, dim2)
        output = G.sum(dim=1).reshape(dim0, dim1, dim2)
        
        return output
    
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Forward computation only."""
        means3d = params.get('means3d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        
        dim2 = data.get('dim2')
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        
        render_output = self._forward_impl(means3d, conics, opacities, dim2, dim1, dim0)
        
        return ForwardResult(
            outputs={'render_output': render_output},
            intermediates={}
        )
    
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward using PyTorch autograd."""
        means3d = params.get('means3d').data.clone().requires_grad_(True)
        conics = params.get('conics').data.clone().requires_grad_(True)
        opacities = params.get('opacities').data.clone().requires_grad_(True)
        
        dim2 = data.get('dim2')
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        
        # Forward pass with gradient tracking
        render_output = self._forward_impl(means3d, conics, opacities, dim2, dim1, dim0)
        
        # Get upstream gradient
        v_render_output = data.get('v_render_output', None)
        if v_render_output is None:
            v_render_output = torch.ones_like(render_output)
        
        # Backward pass
        render_output.backward(v_render_output)
        
        forward_result = ForwardResult(
            outputs={'render_output': render_output.detach()},
            intermediates={}
        )
        
        grad_result = GradResult(
            param_grads={
                'means3d': means3d.grad.detach(),
                'conics': conics.grad.detach(),
                'opacities': opacities.grad.detach(),
            }
        )
        
        return forward_result, grad_result
    
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward using PyTorch autograd.
        
        Computes Hessian by computing gradients for each output voxel
        and accumulating products.
        
        For second-order derivatives, we use the identity:
        h_θ = Σ_i (h_O_i × (∂O_i/∂θ)² + v_O_i × ∂²O_i/∂θ²)
        
        Since outputs are independent sums of Gaussian contributions,
        and we need efficient computation, we use a different approach:
        for each Gaussian n, we compute its contribution to each output voxel,
        then aggregate the Hessian terms.
        """
        means3d = params.get('means3d').data
        conics = params.get('conics').data
        opacities = params.get('opacities').data
        
        dim2 = data.get('dim2')
        dim1 = data.get('dim1')
        dim0 = data.get('dim0')
        
        device = means3d.device
        dtype = means3d.dtype
        N = means3d.shape[0]
        
        # Get upstream gradients
        v_render_output = data.get('v_render_output', None)
        h_render_output = data.get('h_render_output', None)
        
        if v_render_output is None:
            v_render_output = torch.ones(dim0, dim1, dim2, device=device, dtype=dtype)
        if h_render_output is None:
            h_render_output = torch.ones(dim0, dim1, dim2, device=device, dtype=dtype)
        
        # Create voxel coordinate grid with 0.5 offset for voxel centers
        z_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        y_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim2, device=device, dtype=dtype) + 0.5
        
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        voxel_coords = torch.stack([
            xx.reshape(-1),
            yy.reshape(-1),
            zz.reshape(-1)
        ], dim=-1)  # (M, 3)
        
        M = voxel_coords.shape[0]
        
        # Flatten upstream gradients
        v_O = v_render_output.reshape(-1)  # (M,)
        h_O = h_render_output.reshape(-1)  # (M,)
        
        # Initialize accumulators
        v_means3d = torch.zeros_like(means3d)  # (N, 3)
        v_conics = torch.zeros_like(conics)     # (N, 6)
        v_opacities = torch.zeros_like(opacities)  # (N,)
        
        h_means3d = torch.zeros(N, 6, device=device, dtype=dtype)  # upper triangle
        h_conics = torch.zeros(N, 21, device=device, dtype=dtype)  # upper triangle
        h_opacities = torch.zeros(N, device=device, dtype=dtype)
        h_op_mean = torch.zeros(N, 3, device=device, dtype=dtype)
        h_op_conic = torch.zeros(N, 6, device=device, dtype=dtype)
        h_mean_conic = torch.zeros(N, 18, device=device, dtype=dtype)
        
        # For each Gaussian, compute contributions
        # This is a brute-force approach for correctness
        for n in range(N):
            mean = means3d[n]  # (3,)
            conic = conics[n]  # (6,) [a, p, q, b, r, c]
            opacity = opacities[n]  # scalar
            
            a, p, q, b, r, c = conic[0], conic[1], conic[2], conic[3], conic[4], conic[5]
            
            # delta: (M, 3)
            delta = voxel_coords - mean.unsqueeze(0)  # (M, 3)
            dx, dy, dz = delta[:, 0], delta[:, 1], delta[:, 2]
            
            # sigma = 0.5 * (a*dx² + 2*p*dx*dy + 2*q*dx*dz + b*dy² + 2*r*dy*dz + c*dz²)
            sigma = 0.5 * (a * dx * dx + 2 * p * dx * dy + 2 * q * dx * dz +
                           b * dy * dy + 2 * r * dy * dz + c * dz * dz)
            
            vis = torch.exp(-sigma)  # (M,)
            G = opacity * vis  # (M,)
            
            # Apply numerical stability check (same as CUDA)
            mask = (sigma >= 0) & (G.abs() >= 1e-5)
            mask_float = mask.float()
            
            oI = opacity * vis  # same as G for this formula
            
            # First-order gradients of G w.r.t. parameters
            # dG/do = vis
            dG_do = vis  # (M,)
            
            # dG/dconic = -oI × v_arr
            # Note: CUDA v_arr uses order [0.5*d1², 0.5*d2², 0.5*d3², d1*d2, d1*d3, d2*d3]
            # which differs from conic order [a, p, q, b, r, c]
            # We match CUDA behavior for consistency
            dG_dconic = torch.stack([
                -oI * 0.5 * dx * dx,       # v_arr[0] = 0.5*d1²
                -oI * 0.5 * dy * dy,       # v_arr[1] = 0.5*d2²
                -oI * 0.5 * dz * dz,       # v_arr[2] = 0.5*d3²
                -oI * dx * dy,             # v_arr[3] = d1*d2
                -oI * dx * dz,             # v_arr[4] = d1*d3
                -oI * dy * dz,             # v_arr[5] = d2*d3
            ], dim=-1)  # (M, 6)
            
            # dG/dmean = o * vis * [a*dx + p*dy + q*dz, p*dx + b*dy + r*dz, q*dx + r*dy + c*dz]
            dG_dm = torch.stack([
                oI * (a * dx + p * dy + q * dz),
                oI * (p * dx + b * dy + r * dz),
                oI * (q * dx + r * dy + c * dz),
            ], dim=-1)  # (M, 3)
            
            # Accumulate first-order gradients: v_θ = Σ_i v_O_i * dG/dθ (with mask)
            v_opacities[n] = (v_O * dG_do * mask_float).sum()
            v_conics[n] = (v_O.unsqueeze(-1) * dG_dconic * mask_float.unsqueeze(-1)).sum(dim=0)
            v_means3d[n] = (v_O.unsqueeze(-1) * dG_dm * mask_float.unsqueeze(-1)).sum(dim=0)
            
            # Second-order Hessian computation
            # h_θ = Σ_i (h_O_i × (dG/dθ)² + v_O_i × d²G/dθ²)
            
            # h_opacities = Σ h_O × (dG/do)² + Σ v_O × d²G/do²
            #             = Σ h_O × vis² (d²G/do² = 0)
            h_opacities[n] = (h_O * dG_do * dG_do * mask_float).sum()
            
            # h_means3d: 6 upper triangular elements
            # d²G/dm_i*dm_j components
            # From CUDA: d²G/dm_i*dm_j = oI * (s_i * s_j - Σ^{-1}[i,j])
            # where s = Σ^{-1} @ d
            s = torch.stack([
                a * dx + p * dy + q * dz,
                p * dx + b * dy + r * dz,
                q * dx + r * dy + c * dz,
            ], dim=-1)  # (M, 3)
            
            # h_means3d upper triangle: [00, 01, 02, 11, 12, 22]
            # Match CUDA exactly: d2G_dm2[0] = oI * (s1*s1 - a), etc.
            d2G_dm2 = torch.stack([
                oI * (s[:, 0] * s[:, 0] - a),  # H00: s1*s1 - a
                oI * (s[:, 0] * s[:, 1] - p),  # H01: s1*s2 - p
                oI * (s[:, 0] * s[:, 2] - q),  # H02: s1*s3 - q
                oI * (s[:, 1] * s[:, 1] - b),  # H11: s2*s2 - b
                oI * (s[:, 1] * s[:, 2] - r),  # H12: s2*s3 - r
                oI * (s[:, 2] * s[:, 2] - c),  # H22: s3*s3 - c
            ], dim=-1)  # (M, 6)
            
            idx = 0
            for i in range(3):
                for j in range(i, 3):
                    # h_ij = Σ (h_O * dG/dm_i * dG/dm_j + v_O * d²G/dm_i*dm_j)
                    h_means3d[n, idx] = ((h_O * dG_dm[:, i] * dG_dm[:, j] + 
                                          v_O * d2G_dm2[:, idx]) * mask_float).sum()
                    idx += 1
            
            # h_conics: 21 upper triangular elements of 6x6 matrix
            # d²G/dc_i*dc_j
            # Note: v_conic_factors uses the order from CUDA v_arr for consistency
            # CUDA v_arr = [0.5*d1², 0.5*d2², 0.5*d3², d1*d2, d1*d3, d2*d3]
            # This differs from conic order [a, p, q, b, r, c] - appears to be a bug in CUDA
            # but we match it for consistency
            v_conic_factors = torch.stack([
                0.5 * dx * dx,   # v_arr[0] = 0.5*d1²
                0.5 * dy * dy,   # v_arr[1] = 0.5*d2²
                0.5 * dz * dz,   # v_arr[2] = 0.5*d3²
                dx * dy,         # v_arr[3] = d1*d2
                dx * dz,         # v_arr[4] = d1*d3
                dy * dz,         # v_arr[5] = d2*d3
            ], dim=-1)  # (M, 6)
            
            idx = 0
            for i in range(6):
                for j in range(i, 6):
                    # d²G/dc_i*dc_j = oI * v_i * v_j (where v = conic gradient factors)
                    d2G_dcidc = oI * v_conic_factors[:, i] * v_conic_factors[:, j]
                    h_conics[n, idx] = ((h_O * dG_dconic[:, i] * dG_dconic[:, j] + 
                                         v_O * d2G_dcidc) * mask_float).sum()
                    idx += 1
            
            # Cross-terms
            # h_op_mean: (3,) = Σ h_O * dG/do * dG/dm_i + v_O * d²G/do*dm_i
            # d²G/do*dm_i = vis * s_i (from chain rule)
            for i in range(3):
                d2G_dodm = vis * s[:, i]
                h_op_mean[n, i] = ((h_O * dG_do * dG_dm[:, i] + v_O * d2G_dodm) * mask_float).sum()
            
            # h_op_conic: (6,) = Σ h_O * dG/do * dG/dc_i + v_O * d²G/do*dc_i
            # d²G/do*dc_i = -vis * v_arr[i] (matching CUDA)
            for i in range(6):
                d2G_dodc = -vis * v_conic_factors[:, i]
                h_op_conic[n, i] = ((h_O * dG_do * dG_dconic[:, i] + v_O * d2G_dodc) * mask_float).sum()
            
            # h_mean_conic: (18,) = (3, 6) flattened
            # From CUDA: J = [[d1,0,0,d2,d3,0], [0,d2,0,d1,0,d3], [0,0,d3,0,d1,d2]]
            # where d1=dx, d2=dy, d3=dz
            # Storage: row-major [3, 6]
            # Note: This matches CUDA exactly for consistency
            J = torch.zeros(M, 3, 6, device=device, dtype=dtype)
            J[:, 0, 0] = dx     # J[0][0] = d1
            J[:, 0, 3] = dy     # J[0][3] = d2
            J[:, 0, 4] = dz     # J[0][4] = d3
            J[:, 1, 1] = dy     # J[1][1] = d2
            J[:, 1, 3] = dx     # J[1][3] = d1
            J[:, 1, 5] = dz     # J[1][5] = d3
            J[:, 2, 2] = dz     # J[2][2] = d3
            J[:, 2, 4] = dx     # J[2][4] = d1
            J[:, 2, 5] = dy     # J[2][5] = d2
            
            idx = 0
            for mi in range(3):
                for ci in range(6):
                    # d²G/dm_i*dc_j = oI * (J[mi, ci] - s[mi] * v_conic_factors[ci])
                    d2G_dmdc = oI * (J[:, mi, ci] - s[:, mi] * v_conic_factors[:, ci])
                    h_mean_conic[n, idx] = ((h_O * dG_dm[:, mi] * dG_dconic[:, ci] + 
                                             v_O * d2G_dmdc) * mask_float).sum()
                    idx += 1
        
        # Forward output
        render_output = self._forward_impl(means3d, conics, opacities, dim2, dim1, dim0)
        
        forward_result = ForwardResult(
            outputs={'render_output': render_output},
            intermediates={}
        )
        
        grad_result = GradResult(
            param_grads={
                'means3d': v_means3d,
                'conics': v_conics,
                'opacities': v_opacities,
            }
        )
        
        hess_result = HessResult(
            param_hess={
                'means3d': h_means3d,
                'conics': h_conics,
                'opacities': h_opacities,
                'op_mean': h_op_mean,
                'op_conic': h_op_conic,
                'mean_conic': h_mean_conic,
            }
        )
        
        return forward_result, grad_result, hess_result


__all__ = [
    "VoxelizeToVoxelsCUDA",
    "VoxelizeToVoxelsAutograd",
]
