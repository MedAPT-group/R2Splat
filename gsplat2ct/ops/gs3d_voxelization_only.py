"""
3DGS Voxelization-Only Pipeline: End-to-end from raw Gaussian parameters to volume loss.

This module chains 4 operators into an end-to-end pipeline:
    ParameterTransform → WorldToVolumeMapping → VoxelizeToVoxels → L2Loss

Pipeline:
    (means, quats, scale_ori, opacity)
        → [ParameterTransform]          → (covars, max_scales)
        → [WorldToVolumeMapping]        → (means3d_v, conics, radii)
        → [VoxelizeToVoxels]            → render_output (dim0, dim1, dim2)
        → [L2Loss]                      → loss (scalar)

Outer (optimizable) parameters:
    - means:     (N, 3)  world-space Gaussian centres
    - quats:     (N, 4)  unit quaternions (w, x, y, z)
    - scale_ori: (N, 3)  raw scale parameters (before sigmoid)
    - opacity:   (N,)    per-Gaussian opacity

Note: Unlike the 2D projection pipeline, there is NO camera batch dimension.
      Volume reconstruction uses a single volume target.

Second-order backward computes:
    - Gradients for all 4 parameter groups
    - Self-Hessians for each parameter group
    - Cross-Hessians between all pairs

Hessian chain rule (backward through the chain):
    H_theta = J^T H_phi J + sum_k v_k Hess_k(theta)

Author: Dy164
Date: 2026-02-23
"""

import math
import torch
from typing import Optional, Tuple
from .base import SecondForwardBackwardBase
from .data import Params, Data, Config, ForwardResult, GradResult, HessResult


# ============================================================================
# Helper: ParamWrapper
# ============================================================================

class _ParamWrapper:
    """Simple wrapper to satisfy ParameterLike protocol."""
    def __init__(self, data: torch.Tensor):
        self.data = data


# ============================================================================
# CUDA Pipeline Implementation
# ============================================================================

class GS3DVoxelizationOnlyCUDA(SecondForwardBackwardBase):
    """
    CUDA-accelerated end-to-end voxelization pipeline:
        ParameterTransform → WorldToVolumeMapping → VoxelizeToVoxels → L2Loss

    Expected params:
        - means:     (N, 3)  world-space Gaussian centres
        - quats:     (N, 4)  unit quaternions
        - scale_ori: (N, 3)  raw scale params
        - opacity:   (N,)    per-Gaussian opacity

    Expected data:
        - ranges:  (3,) physical extent per axis
        - centers: (3,) physical center per axis
        - target:  (dim0, dim1, dim2) ground-truth volume for loss

    Expected config:
        - dim2: int  (voxel grid x, fastest)
        - dim1: int  (voxel grid y, middle)
        - dim0: int  (voxel grid z, slowest)
        - tile_size:    int   (default 4)
        - eps3d:        float (default 1e-4)
        - radius_clip:  float (default 3.0)
        - scale_min:    float (default 0.01)
        - scale_max:    float (default 2.0)
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'quats', 'scale_ori', 'opacity']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        means = params.get('means').data
        if not means.is_cuda:
            raise ValueError("Parameters must be CUDA tensors")
        self.initialized = True

    # ------------------------------------------------------------------
    # forward_only
    # ------------------------------------------------------------------
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        """Run the full forward pipeline and return loss."""
        from .parameter_transform import ParameterTransformCUDA
        from .world_to_volume_mapping import WorldToVolumeMappingCUDA
        from .voxelize_to_voxels import VoxelizeToVoxelsCUDA
        from .loss import L2LossCUDA

        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        N = means.shape[0]

        ranges = data.get('ranges')
        centers = data.get('centers')
        target = data.get('target')
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        tile_size = config.get('tile_size', 4)
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        # --- Stage 0: ParameterTransform ---
        pt_params = Params()
        pt_params.add_param('quats', _ParamWrapper(quats))
        pt_params.add_param('scale_ori', _ParamWrapper(scale_ori))
        pt_config = Config()
        pt_config.set('scale_min', scale_min)
        pt_config.set('scale_max', scale_max)

        pt_op = ParameterTransformCUDA()
        pt_op.initialize(pt_params, pt_config)
        pt_fwd = pt_op.forward_only(pt_params, Data(), pt_config)
        covars = pt_fwd.outputs['covars']          # (N, 6)
        max_scales = pt_fwd.outputs['max_scales']  # (N,)

        # --- Stage 1: WorldToVolumeMapping ---
        wvm_params = Params()
        wvm_params.add_param('means', _ParamWrapper(means))
        wvm_params.add_param('covars', _ParamWrapper(covars))
        wvm_params.add_param('max_scales', _ParamWrapper(max_scales))
        wvm_data = Data()
        wvm_data.set('ranges', ranges)
        wvm_data.set('centers', centers)
        wvm_config = Config()
        wvm_config.set('dim2', dim2)
        wvm_config.set('dim1', dim1)
        wvm_config.set('dim0', dim0)
        wvm_config.set('eps3d', eps3d)
        wvm_config.set('radius_clip', radius_clip)

        wvm_op = WorldToVolumeMappingCUDA()
        wvm_op.initialize(wvm_params, wvm_config)
        wvm_fwd = wvm_op.forward_only(wvm_params, wvm_data, wvm_config)
        means3d_v = wvm_fwd.outputs['means3d_v']  # (N, 3)
        conics = wvm_fwd.outputs['conics']          # (N, 6)
        radii = wvm_fwd.outputs['radii']            # (N, 3)

        # --- Stage 2: VoxelizeToVoxels ---
        vox_params = Params()
        vox_params.add_param('means3d', _ParamWrapper(means3d_v))
        vox_params.add_param('conics', _ParamWrapper(conics))
        vox_params.add_param('opacities', _ParamWrapper(opacity))
        vox_data = Data()
        vox_data.set('radii', radii)
        vox_data.set('dim2', dim2)
        vox_data.set('dim1', dim1)
        vox_data.set('dim0', dim0)
        vox_data.set('tile_size', tile_size)

        vox_op = VoxelizeToVoxelsCUDA()
        vox_op.initialize(vox_params)
        vox_fwd = vox_op.forward_only(vox_params, vox_data)
        render_output = vox_fwd.outputs['render_output']  # (dim0, dim1, dim2)

        # --- Stage 3: L2 Loss ---
        loss_params = Params()
        loss_params.add_param('prediction', _ParamWrapper(render_output))
        loss_data = Data()
        loss_data.set('target', target)

        loss_op = L2LossCUDA()
        loss_op.initialize(loss_params)
        loss_fwd = loss_op.forward_only(loss_params, loss_data)

        return ForwardResult(
            outputs={
                'loss': loss_fwd.outputs['loss'],
                'render_output': render_output,
                'means3d_v': means3d_v,
                'conics': conics,
                'covars': covars,
            },
            intermediates={}
        )

    # ------------------------------------------------------------------
    # forward_backward1
    # ------------------------------------------------------------------
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        """Forward + first-order backward through the full 4-stage pipeline."""
        from .parameter_transform import ParameterTransformCUDA
        from .world_to_volume_mapping import WorldToVolumeMappingCUDA
        from .voxelize_to_voxels import VoxelizeToVoxelsCUDA
        from .loss import L2LossCUDA

        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        N = means.shape[0]

        ranges = data.get('ranges')
        centers = data.get('centers')
        target = data.get('target')
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        tile_size = config.get('tile_size', 4)
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        # ============== FORWARD PASS ==============

        # Stage 0: ParameterTransform
        pt_params = Params()
        pt_params.add_param('quats', _ParamWrapper(quats))
        pt_params.add_param('scale_ori', _ParamWrapper(scale_ori))
        pt_config = Config()
        pt_config.set('scale_min', scale_min)
        pt_config.set('scale_max', scale_max)

        pt_op = ParameterTransformCUDA()
        pt_op.initialize(pt_params, pt_config)
        pt_fwd = pt_op.forward_only(pt_params, Data(), pt_config)
        covars = pt_fwd.outputs['covars']
        max_scales = pt_fwd.outputs['max_scales']

        # Stage 1: WorldToVolumeMapping
        wvm_params = Params()
        wvm_params.add_param('means', _ParamWrapper(means))
        wvm_params.add_param('covars', _ParamWrapper(covars))
        wvm_params.add_param('max_scales', _ParamWrapper(max_scales))
        wvm_data = Data()
        wvm_data.set('ranges', ranges)
        wvm_data.set('centers', centers)
        wvm_config = Config()
        wvm_config.set('dim2', dim2)
        wvm_config.set('dim1', dim1)
        wvm_config.set('dim0', dim0)
        wvm_config.set('eps3d', eps3d)
        wvm_config.set('radius_clip', radius_clip)

        wvm_op = WorldToVolumeMappingCUDA()
        wvm_op.initialize(wvm_params, wvm_config)
        wvm_fwd = wvm_op.forward_only(wvm_params, wvm_data, wvm_config)
        means3d_v = wvm_fwd.outputs['means3d_v']
        conics = wvm_fwd.outputs['conics']
        radii = wvm_fwd.outputs['radii']

        # Stage 2: VoxelizeToVoxels
        vox_params = Params()
        vox_params.add_param('means3d', _ParamWrapper(means3d_v))
        vox_params.add_param('conics', _ParamWrapper(conics))
        vox_params.add_param('opacities', _ParamWrapper(opacity))
        vox_data = Data()
        vox_data.set('radii', radii)
        vox_data.set('dim2', dim2)
        vox_data.set('dim1', dim1)
        vox_data.set('dim0', dim0)
        vox_data.set('tile_size', tile_size)

        vox_op = VoxelizeToVoxelsCUDA()
        vox_op.initialize(vox_params)
        vox_fwd = vox_op.forward_only(vox_params, vox_data)
        render_output = vox_fwd.outputs['render_output']

        # Stage 3: L2 Loss
        loss_params = Params()
        loss_params.add_param('prediction', _ParamWrapper(render_output))
        loss_data = Data()
        loss_data.set('target', target)

        loss_op = L2LossCUDA()
        loss_op.initialize(loss_params)
        loss_fwd, loss_grad = loss_op.forward_backward1(loss_params, loss_data)
        v_render_output = loss_grad.param_grads['prediction']  # (dim0, dim1, dim2)

        # ============== BACKWARD PASS ==============

        # Stage 2 bwd: VoxelizeToVoxels
        vox_data_bwd = Data()
        vox_data_bwd.set('radii', radii)
        vox_data_bwd.set('dim2', dim2)
        vox_data_bwd.set('dim1', dim1)
        vox_data_bwd.set('dim0', dim0)
        vox_data_bwd.set('tile_size', tile_size)
        vox_data_bwd.set('v_render_output', v_render_output)

        _, vox_grad = vox_op.forward_backward1(vox_params, vox_data_bwd)
        v_means3d_v = vox_grad.param_grads['means3d']    # (N, 3)
        v_conics = vox_grad.param_grads['conics']          # (N, 6)
        v_opacity = vox_grad.param_grads['opacities']      # (N,)

        # Stage 1 bwd: WorldToVolumeMapping
        wvm_data_bwd = Data()
        wvm_data_bwd.set('ranges', ranges)
        wvm_data_bwd.set('centers', centers)
        wvm_data_bwd.set('v_means3d_v', v_means3d_v)
        wvm_data_bwd.set('v_conics', v_conics)

        _, wvm_grad = wvm_op.forward_backward1(wvm_params, wvm_data_bwd, wvm_config)
        v_means = wvm_grad.param_grads['means']    # (N, 3)
        v_covars = wvm_grad.param_grads['covars']  # (N, 6)

        # Stage 0 bwd: ParameterTransform
        pt_data_bwd = Data()
        pt_data_bwd.set('v_covars', v_covars)

        _, pt_grad = pt_op.forward_backward1(pt_params, pt_data_bwd, pt_config)
        v_quats = pt_grad.param_grads['quats']          # (N, 4)
        v_scale_ori = pt_grad.param_grads['scale_ori']  # (N, 3)

        forward_result = ForwardResult(
            outputs={
                'loss': loss_fwd.outputs['loss'],
                'render_output': render_output,
            },
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means,
                'quats': v_quats,
                'scale_ori': v_scale_ori,
                'opacity': v_opacity,
            }
        )
        return forward_result, grad_result

    # ------------------------------------------------------------------
    # forward_backward2
    # ------------------------------------------------------------------
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward through the full 4-stage pipeline.

        Chain rule for Hessians (backward through layers):

        The pipeline is:
            (means, quats, scale_ori, opacity)
                → covars → (means3d_v, conics) → render_output → loss

        Stage 3 (L2Loss) gives v_render and h_render.
        Stage 2 (VoxelizeToVoxels bwd2) gives all Hessian blocks for
            (means3d_v, conics, opacity).
        Stage 1 (WorldToVolumeMapping bwd2) propagates through
            (means, covars) → (means3d_v, conics).
        Stage 0 (ParameterTransform bwd2) propagates through
            (quats, scale_ori) → covars.
        """
        from .parameter_transform import ParameterTransformCUDA
        from .world_to_volume_mapping import WorldToVolumeMappingCUDA
        from .voxelize_to_voxels import VoxelizeToVoxelsCUDA
        from .loss import L2LossCUDA

        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        N = means.shape[0]
        dtype = means.dtype
        device = means.device

        ranges = data.get('ranges')
        centers = data.get('centers')
        target = data.get('target')
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        tile_size = config.get('tile_size', 4)
        eps3d = config.get('eps3d', 1e-4)
        radius_clip = config.get('radius_clip', 3.0)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        # ==================================================================
        # FORWARD PASS: stages 0-3
        # ==================================================================

        # Stage 0: ParameterTransform
        pt_params = Params()
        pt_params.add_param('quats', _ParamWrapper(quats))
        pt_params.add_param('scale_ori', _ParamWrapper(scale_ori))
        pt_config = Config()
        pt_config.set('scale_min', scale_min)
        pt_config.set('scale_max', scale_max)

        pt_op = ParameterTransformCUDA()
        pt_op.initialize(pt_params, pt_config)
        pt_fwd = pt_op.forward_only(pt_params, Data(), pt_config)
        covars = pt_fwd.outputs['covars']
        max_scales = pt_fwd.outputs['max_scales']

        # Stage 1: WorldToVolumeMapping
        wvm_params = Params()
        wvm_params.add_param('means', _ParamWrapper(means))
        wvm_params.add_param('covars', _ParamWrapper(covars))
        wvm_params.add_param('max_scales', _ParamWrapper(max_scales))
        wvm_data = Data()
        wvm_data.set('ranges', ranges)
        wvm_data.set('centers', centers)
        wvm_config = Config()
        wvm_config.set('dim2', dim2)
        wvm_config.set('dim1', dim1)
        wvm_config.set('dim0', dim0)
        wvm_config.set('eps3d', eps3d)
        wvm_config.set('radius_clip', radius_clip)

        wvm_op = WorldToVolumeMappingCUDA()
        wvm_op.initialize(wvm_params, wvm_config)
        wvm_fwd = wvm_op.forward_only(wvm_params, wvm_data, wvm_config)
        means3d_v = wvm_fwd.outputs['means3d_v']
        conics = wvm_fwd.outputs['conics']
        radii = wvm_fwd.outputs['radii']

        # Stage 2: VoxelizeToVoxels
        vox_params = Params()
        vox_params.add_param('means3d', _ParamWrapper(means3d_v))
        vox_params.add_param('conics', _ParamWrapper(conics))
        vox_params.add_param('opacities', _ParamWrapper(opacity))
        vox_data = Data()
        vox_data.set('radii', radii)
        vox_data.set('dim2', dim2)
        vox_data.set('dim1', dim1)
        vox_data.set('dim0', dim0)
        vox_data.set('tile_size', tile_size)

        vox_op = VoxelizeToVoxelsCUDA()
        vox_op.initialize(vox_params)
        vox_fwd = vox_op.forward_only(vox_params, vox_data)
        render_output = vox_fwd.outputs['render_output']

        # Stage 3: L2 Loss
        loss_params = Params()
        loss_params.add_param('prediction', _ParamWrapper(render_output))
        loss_data = Data()
        loss_data.set('target', target)

        loss_op = L2LossCUDA()
        loss_op.initialize(loss_params)
        loss_fwd, loss_grad, loss_hess = loss_op.forward_backward2(
            loss_params, loss_data
        )
        v_render_output = loss_grad.param_grads['prediction']  # (dim0, dim1, dim2)
        h_render_output = loss_hess.param_hess['prediction']   # (dim0, dim1, dim2)

        # ==================================================================
        # BACKWARD PASS stage 2: VoxelizeToVoxels bwd2
        # ==================================================================
        vox_data_bwd2 = Data()
        vox_data_bwd2.set('radii', radii)
        vox_data_bwd2.set('dim2', dim2)
        vox_data_bwd2.set('dim1', dim1)
        vox_data_bwd2.set('dim0', dim0)
        vox_data_bwd2.set('tile_size', tile_size)
        vox_data_bwd2.set('v_render_output', v_render_output)
        vox_data_bwd2.set('h_render_output', h_render_output)

        _, vox_grad, vox_hess = vox_op.forward_backward2(
            vox_params, vox_data_bwd2
        )

        v_means3d_v_vox = vox_grad.param_grads['means3d']    # (N, 3)
        v_conics_vox = vox_grad.param_grads['conics']          # (N, 6)
        v_opacity_vox = vox_grad.param_grads['opacities']      # (N,)

        h_means3d_v_vox = vox_hess.param_hess['means3d']      # (N, 6)
        h_conics_vox = vox_hess.param_hess['conics']            # (N, 21)
        h_opacity_vox = vox_hess.param_hess['opacities']        # (N,)
        h_op_mean_vox = vox_hess.param_hess['op_mean']          # (N, 3)
        h_op_conic_vox = vox_hess.param_hess['op_conic']        # (N, 6)
        h_mean_conic_vox = vox_hess.param_hess['mean_conic']    # (N, 18)

        # ==================================================================
        # BACKWARD PASS stage 1: WorldToVolumeMapping bwd2
        # ==================================================================
        # WorldToVolumeMapping maps (means, covars) → (means3d_v, conics).
        # We need to pass downstream Hessians from voxelization.
        # The cross-term h_means3d_v_conics from voxelization is h_mean_conic_vox: (N, 18)
        # in 3x6 row-major format.

        wvm_data_bwd2 = Data()
        wvm_data_bwd2.set('ranges', ranges)
        wvm_data_bwd2.set('centers', centers)
        wvm_data_bwd2.set('v_means3d_v', v_means3d_v_vox)
        wvm_data_bwd2.set('v_conics', v_conics_vox)
        wvm_data_bwd2.set('h_means3d_v', h_means3d_v_vox)
        wvm_data_bwd2.set('h_conics', h_conics_vox)
        wvm_data_bwd2.set('h_means3d_v_conics', h_mean_conic_vox)
        wvm_data_bwd2.set('h_means3d_v_opacity', h_op_mean_vox)
        wvm_data_bwd2.set('h_conics_opacity', h_op_conic_vox)

        _, wvm_grad, wvm_hess = wvm_op.forward_backward2(
            wvm_params, wvm_data_bwd2, wvm_config
        )

        v_means = wvm_grad.param_grads['means']    # (N, 3)
        v_covars = wvm_grad.param_grads['covars']  # (N, 6)

        h_means = wvm_hess.param_hess['means']               # (N, 6)
        h_covars = wvm_hess.param_hess['covars']             # (N, 21)
        h_means_covars = wvm_hess.param_hess['means_covars'] # (N, 18)
        h_means_opacity = wvm_hess.param_hess['means_opacity']   # (N, 3)
        h_covars_opacity = wvm_hess.param_hess['covars_opacity'] # (N, 6)

        # ==================================================================
        # BACKWARD PASS stage 0: ParameterTransform bwd2
        # ==================================================================
        pt_data_bwd2 = Data()
        pt_data_bwd2.set('v_covars', v_covars)
        pt_data_bwd2.set('h_covars', h_covars)
        # h_means_covars is H_{means, covars} in 3x6 row-major (N, 18).
        # ParameterTransform expects h_covars_means = H_{covars, means} in 6x3 row-major.
        # Transpose: 3x6 → 6x3
        h_mc_mat = h_means_covars.reshape(N, 3, 6)
        h_cm_mat = h_mc_mat.transpose(-1, -2)  # (N, 6, 3)
        h_covars_means_for_pt = h_cm_mat.reshape(N, 18)
        pt_data_bwd2.set('h_covars_means', h_covars_means_for_pt)
        pt_data_bwd2.set('h_covars_opacity', h_covars_opacity)

        _, pt_grad, pt_hess = pt_op.forward_backward2(
            pt_params, pt_data_bwd2, pt_config
        )

        v_quats = pt_grad.param_grads['quats']           # (N, 4)
        v_scale_ori = pt_grad.param_grads['scale_ori']    # (N, 3)

        h_quats = pt_hess.param_hess['quats']             # (N, 10)
        h_scale_ori = pt_hess.param_hess['scale_ori']     # (N, 3)
        h_quats_scale_ori = pt_hess.cross_hess['quats_scale_ori']   # (N, 12)
        h_quats_means = pt_hess.cross_hess['quats_means']           # (N, 12)
        h_quats_opacity = pt_hess.cross_hess['quats_opacity']       # (N, 4)
        h_scale_ori_means = pt_hess.cross_hess['scale_ori_means']   # (N, 9)
        h_scale_ori_opacity = pt_hess.cross_hess['scale_ori_opacity'] # (N, 3)

        # ==================================================================
        # Assemble results
        # ==================================================================
        forward_result = ForwardResult(
            outputs={
                'loss': loss_fwd.outputs['loss'],
                'render_output': render_output,
            },
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means,
                'quats': v_quats,
                'scale_ori': v_scale_ori,
                'opacity': v_opacity_vox,
            }
        )
        hess_result = HessResult(
            param_hess={
                'means': h_means,                   # (N, 6)
                'quats': h_quats,                    # (N, 10)
                'scale_ori': h_scale_ori,            # (N, 3)
                'opacity': h_opacity_vox,            # (N,)
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori,        # (N, 12)
                'means_opacity': h_means_opacity,            # (N, 3)
                'quats_means': h_quats_means,                # (N, 12)
                'quats_opacity': h_quats_opacity,            # (N, 4)
                'scale_ori_means': h_scale_ori_means,        # (N, 9)
                'scale_ori_opacity': h_scale_ori_opacity,    # (N, 3)
            }
        )
        return forward_result, grad_result, hess_result


# ============================================================================
# Autograd Pipeline Implementation
# ============================================================================

class GS3DVoxelizationOnlyAutograd(SecondForwardBackwardBase):
    """
    Pure PyTorch autograd-based end-to-end voxelization pipeline (ground-truth reference).

    Uses PyTorch automatic differentiation to compute first-order and
    second-order derivatives of the full 4-stage pipeline:
        means, quats, scale_ori, opacity → loss

    Same params/data/config interface as GS3DVoxelizationOnlyCUDA.

    Key design: voxels are decoupled in the CUDA implementation (each voxel
    is computed independently from other voxels). To match this behavior,
    the autograd second-order backward computes per-Gaussian Hessian blocks
    without cross-voxel second-order terms.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, params: Params, config: Optional[Config] = None) -> None:
        for name in ['means', 'quats', 'scale_ori', 'opacity']:
            if params.get(name) is None:
                raise ValueError(f"Missing required parameter: {name}")
        self.initialized = True

    def _forward_pipeline(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scale_ori: torch.Tensor,
        opacity: torch.Tensor,
        ranges: torch.Tensor,
        centers: torch.Tensor,
        target: torch.Tensor,
        dim2: int,
        dim1: int,
        dim0: int,
        eps3d: float,
        scale_min: float,
        scale_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the full differentiable pipeline in pure PyTorch.

        Returns:
            loss: scalar
            render_output: (dim0, dim1, dim2)
        """
        from .parameter_transform import _compute_covars
        from .world_to_volume_mapping import _compute_world_to_volume_mapping

        N = means.shape[0]
        device = means.device
        dtype = means.dtype

        # Stage 0: ParameterTransform
        covars = _compute_covars(quats, scale_ori, scale_min, scale_max)  # (N, 6)

        # Compute max_scales for radii (non-differentiable)
        sig = torch.sigmoid(scale_ori)
        scales = sig * (scale_max - scale_min) + scale_min
        max_scales = scales.max(dim=-1).values  # (N,)

        # Stage 1: WorldToVolumeMapping
        means3d_v, conics, radii = _compute_world_to_volume_mapping(
            means, covars, max_scales, ranges, centers,
            dim2, dim1, dim0, eps3d,
        )

        # Stage 2: VoxelizeToVoxels (brute-force vectorized)
        # Create voxel coordinate grid
        z_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
        y_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(dim2, device=device, dtype=dtype) + 0.5

        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        voxel_coords = torch.stack([
            xx.reshape(-1),
            yy.reshape(-1),
            zz.reshape(-1)
        ], dim=-1)  # (M, 3) with [x, y, z]

        # Compute Gaussian contribution at each voxel
        # delta: (M, N, 3)
        delta = voxel_coords.unsqueeze(1) - means3d_v.unsqueeze(0)
        dx = delta[..., 0]
        dy = delta[..., 1]
        dz = delta[..., 2]

        # conics: (N, 6) = [a, p, q, b, r, c]
        a = conics[:, 0].unsqueeze(0)
        p = conics[:, 1].unsqueeze(0)
        q = conics[:, 2].unsqueeze(0)
        b = conics[:, 3].unsqueeze(0)
        r = conics[:, 4].unsqueeze(0)
        c = conics[:, 5].unsqueeze(0)

        sigma = 0.5 * (a * dx * dx + b * dy * dy + c * dz * dz) + \
                p * dx * dy + q * dx * dz + r * dy * dz

        vis = torch.exp(-sigma)
        G = opacity.unsqueeze(0) * vis  # (M, N)

        # Apply numerical stability mask (same as CUDA)
        mask = (sigma >= 0) & (G.abs() >= 1e-5)
        G_masked = torch.where(mask, G, torch.zeros_like(G))

        render_output = G_masked.sum(dim=1).reshape(dim0, dim1, dim2)

        # Stage 3: L2 Loss
        loss = ((render_output - target) ** 2).mean()

        return loss, render_output

    # ------------------------------------------------------------------
    # forward_only
    # ------------------------------------------------------------------
    def forward_only(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> ForwardResult:
        means = params.get('means').data
        quats = params.get('quats').data
        scale_ori = params.get('scale_ori').data
        opacity = params.get('opacity').data
        ranges = data.get('ranges')
        centers = data.get('centers')
        target = data.get('target')
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps3d = config.get('eps3d', 1e-4)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        with torch.no_grad():
            loss, render_output = self._forward_pipeline(
                means, quats, scale_ori, opacity,
                ranges, centers, target,
                dim2, dim1, dim0, eps3d,
                scale_min, scale_max,
            )

        return ForwardResult(
            outputs={'loss': loss, 'render_output': render_output},
            intermediates={}
        )

    # ------------------------------------------------------------------
    # forward_backward1
    # ------------------------------------------------------------------
    def forward_backward1(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult]:
        means_orig = params.get('means').data
        quats_orig = params.get('quats').data
        scale_ori_orig = params.get('scale_ori').data
        opacity_orig = params.get('opacity').data
        ranges = data.get('ranges')
        centers = data.get('centers')
        target = data.get('target')
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps3d = config.get('eps3d', 1e-4)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)

        means = means_orig.detach().clone().requires_grad_(True)
        quats = quats_orig.detach().clone().requires_grad_(True)
        scale_ori = scale_ori_orig.detach().clone().requires_grad_(True)
        opacity = opacity_orig.detach().clone().requires_grad_(True)

        loss, render_output = self._forward_pipeline(
            means, quats, scale_ori, opacity,
            ranges, centers, target,
            dim2, dim1, dim0, eps3d,
            scale_min, scale_max,
        )

        loss.backward()

        forward_result = ForwardResult(
            outputs={'loss': loss.detach(), 'render_output': render_output.detach()},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': means.grad.detach(),
                'quats': quats.grad.detach(),
                'scale_ori': scale_ori.grad.detach(),
                'opacity': opacity.grad.detach(),
            }
        )
        return forward_result, grad_result

    # ------------------------------------------------------------------
    # forward_backward2
    # ------------------------------------------------------------------
    def forward_backward2(
        self,
        params: Params,
        data: Data,
        config: Optional[Config] = None,
    ) -> Tuple[ForwardResult, GradResult, HessResult]:
        """Forward + second-order backward via double autograd.

        Computes full per-Gaussian Hessian blocks by differentiating
        v_theta = dL/d(theta) component by component.

        Note: All parameters are shared (no camera batch dimension).
        opacity is (N,) not (C, N).
        """
        means_orig = params.get('means').data
        quats_orig = params.get('quats').data
        scale_ori_orig = params.get('scale_ori').data
        opacity_orig = params.get('opacity').data
        ranges = data.get('ranges')
        centers = data.get('centers')
        target = data.get('target')
        dim2 = config.get('dim2')
        dim1 = config.get('dim1')
        dim0 = config.get('dim0')
        eps3d = config.get('eps3d', 1e-4)
        scale_min = config.get('scale_min', 0.01)
        scale_max = config.get('scale_max', 2.0)
        N = means_orig.shape[0]
        dtype = means_orig.dtype
        device = means_orig.device

        # Create leaf tensors for double backward
        means = means_orig.detach().clone().requires_grad_(True)
        quats = quats_orig.detach().clone().requires_grad_(True)
        scale_ori = scale_ori_orig.detach().clone().requires_grad_(True)
        opacity = opacity_orig.detach().clone().requires_grad_(True)

        loss, render_output = self._forward_pipeline(
            means, quats, scale_ori, opacity,
            ranges, centers, target,
            dim2, dim1, dim0, eps3d,
            scale_min, scale_max,
        )

        # First backward with create_graph=True
        v_means_g, v_quats_g, v_scale_ori_g, v_opacity_g = torch.autograd.grad(
            outputs=loss,
            inputs=[means, quats, scale_ori, opacity],
            create_graph=True,
            retain_graph=True,
        )

        # ================================================================
        # Compute per-Gaussian Hessian blocks
        # ================================================================

        # means: (N, 3)
        h_pp = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        # quats: (N, 4)
        h_qq = torch.zeros(N, 4, 4, dtype=dtype, device=device)
        # scale_ori: (N, 3)
        h_ss = torch.zeros(N, 3, 3, dtype=dtype, device=device)
        # opacity: (N,)
        h_oo = torch.zeros(N, dtype=dtype, device=device)

        # Cross-Hessians
        h_pq = torch.zeros(N, 3, 4, dtype=dtype, device=device)     # means × quats
        h_ps = torch.zeros(N, 3, 3, dtype=dtype, device=device)     # means × scale_ori
        h_po = torch.zeros(N, 3, dtype=dtype, device=device)        # means × opacity
        h_qs = torch.zeros(N, 4, 3, dtype=dtype, device=device)     # quats × scale_ori
        h_qo = torch.zeros(N, 4, dtype=dtype, device=device)        # quats × opacity
        h_so = torch.zeros(N, 3, dtype=dtype, device=device)        # scale_ori × opacity

        for g in range(N):
            # -- means[g,:] Hessian --
            for i in range(3):
                grads = torch.autograd.grad(
                    outputs=v_means_g[g, i],
                    inputs=[means, quats, scale_ori, opacity],
                    retain_graph=True,
                    allow_unused=True,
                )
                g_p = grads[0] if grads[0] is not None else torch.zeros_like(means)
                g_q = grads[1] if grads[1] is not None else torch.zeros_like(quats)
                g_s = grads[2] if grads[2] is not None else torch.zeros_like(scale_ori)
                g_o = grads[3] if grads[3] is not None else torch.zeros_like(opacity)

                h_pp[g, i, :] = g_p[g, :]
                h_pq[g, i, :] = g_q[g, :]
                h_ps[g, i, :] = g_s[g, :]
                h_po[g, i] = g_o[g]

            # -- quats[g,:] Hessian --
            for i in range(4):
                grads = torch.autograd.grad(
                    outputs=v_quats_g[g, i],
                    inputs=[quats, scale_ori, opacity],
                    retain_graph=True,
                    allow_unused=True,
                )
                g_q = grads[0] if grads[0] is not None else torch.zeros_like(quats)
                g_s = grads[1] if grads[1] is not None else torch.zeros_like(scale_ori)
                g_o = grads[2] if grads[2] is not None else torch.zeros_like(opacity)

                h_qq[g, i, :] = g_q[g, :]
                h_qs[g, i, :] = g_s[g, :]
                h_qo[g, i] = g_o[g]

            # -- scale_ori[g,:] Hessian --
            for i in range(3):
                grads = torch.autograd.grad(
                    outputs=v_scale_ori_g[g, i],
                    inputs=[scale_ori, opacity],
                    retain_graph=True,
                    allow_unused=True,
                )
                g_s = grads[0] if grads[0] is not None else torch.zeros_like(scale_ori)
                g_o = grads[1] if grads[1] is not None else torch.zeros_like(opacity)

                h_ss[g, i, :] = g_s[g, :]
                h_so[g, i] = g_o[g]

            # -- opacity[g] Hessian --
            grads = torch.autograd.grad(
                outputs=v_opacity_g[g],
                inputs=[opacity],
                retain_graph=True,
                allow_unused=True,
            )
            g_o = grads[0] if grads[0] is not None else torch.zeros_like(opacity)
            h_oo[g] = g_o[g]

        # ================================================================
        # Pack outputs into the same format as CUDA
        # ================================================================

        # h_means: upper-tri 3x3 → 6 elements
        h_means_out = torch.zeros(N, 6, dtype=dtype, device=device)
        idx = 0
        for i in range(3):
            for j in range(i, 3):
                h_means_out[:, idx] = h_pp[:, i, j]
                idx += 1

        # h_quats: upper-tri 4x4 → 10 elements
        h_quats_out = torch.zeros(N, 10, dtype=dtype, device=device)
        idx = 0
        for i in range(4):
            for j in range(i, 4):
                h_quats_out[:, idx] = h_qq[:, i, j]
                idx += 1

        # h_scale_ori: diagonal of 3x3
        h_scale_ori_out = torch.stack([h_ss[:, i, i] for i in range(3)], dim=-1)

        # h_quats_scale_ori: 4x3 row-major → 12 elements
        h_quats_scale_ori_out = h_qs.reshape(N, 12)

        # h_quats_means: H_{quats, means} 4x3 row-major → 12 elements
        # h_pq is H_{means, quats} (N, 3, 4), transpose to get H_{quats, means} (N, 4, 3)
        h_quats_means_out = h_pq.transpose(-1, -2).reshape(N, 12)

        # h_scale_ori_means: H_{scale_ori, means} 3x3 row-major → 9 elements
        # h_ps is H_{means, scale_ori} (N, 3, 3), transpose to get H_{scale_ori, means} (N, 3, 3)
        h_scale_ori_means_out = h_ps.transpose(-1, -2).reshape(N, 9)

        # h_means_opacity: (N, 3)
        h_means_opacity_out = h_po

        forward_result = ForwardResult(
            outputs={'loss': loss.detach(), 'render_output': render_output.detach()},
            intermediates={}
        )
        grad_result = GradResult(
            param_grads={
                'means': v_means_g.detach(),
                'quats': v_quats_g.detach(),
                'scale_ori': v_scale_ori_g.detach(),
                'opacity': v_opacity_g.detach(),
            }
        )
        hess_result = HessResult(
            param_hess={
                'means': h_means_out.detach(),
                'quats': h_quats_out.detach(),
                'scale_ori': h_scale_ori_out.detach(),
                'opacity': h_oo.detach(),
            },
            cross_hess={
                'quats_scale_ori': h_quats_scale_ori_out.detach(),
                'means_opacity': h_means_opacity_out.detach(),
                'quats_means': h_quats_means_out.detach(),
                'quats_opacity': h_qo.detach(),
                'scale_ori_means': h_scale_ori_means_out.detach(),
                'scale_ori_opacity': h_so.detach(),
            }
        )
        return forward_result, grad_result, hess_result


# ============================================================================
# Factory Function
# ============================================================================

def create_gs3d_voxelization_only(
    backend: str = "cuda",
) -> SecondForwardBackwardBase:
    """Factory function to create GS3DVoxelizationOnly operator.

    Args:
        backend: "cuda" for CUDA pipeline, "autograd" for PyTorch reference

    Returns:
        GS3DVoxelizationOnly instance
    """
    if backend == "cuda":
        return GS3DVoxelizationOnlyCUDA()
    elif backend == "autograd":
        return GS3DVoxelizationOnlyAutograd()
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'cuda' or 'autograd'."
        )


__all__ = [
    "GS3DVoxelizationOnlyCUDA",
    "GS3DVoxelizationOnlyAutograd",
    "create_gs3d_voxelization_only",
]
