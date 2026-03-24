/**
 * @file WorldToVolumeMapping.h
 * @brief CUDA Kernel Launch Functions for World-to-Volume Mapping
 *
 * This header declares the launch functions that bridge between
 * C++ tensor processing (WorldToVolumeMapping.cpp) and CUDA kernels
 * (WorldToVolumeMappingFwd.cu, WorldToVolumeMappingBwd.cu,
 *  WorldToVolumeMappingBwd2.cu).
 *
 * =========================================================================
 * Module Overview
 * =========================================================================
 *
 * Transforms world-space Gaussian parameters into volume (voxel) space:
 *
 *   World-space inputs:  means [N,3], covars [N,6], max_scales [N]
 *   Volume-space outputs: means3d_v [N,3], conics [N,6], radii [N,3]
 *
 * =========================================================================
 * Forward Pass Pipeline
 * =========================================================================
 *
 * F1. Compute scaling factors (diagonal R matrix):
 *       r_i = dim_i / range_i,  i=0,1,2
 *       R = diag(r2, r1, r0)   (means stored as x,y,z = dim2,dim1,dim0)
 *
 * F2. Affine transform for means:
 *       means3d_v[k] = (means_w[k] - center[k]) * r[k]  + dim[k]/2
 *       where k: 0→x(dim2), 1→y(dim1), 2→z(dim0)
 *
 * F3. Covariance transform (R is diagonal):
 *       Sigma_v = R * Sigma_w * R^T
 *       Since R is diagonal, Sigma_v[i][j] = r_i * r_j * Sigma_w[i][j]
 *
 * F4. Add blur:
 *       Sigma_v[i][i] += eps3d
 *
 * F5. Matrix inversion (3x3 symmetric → conics):
 *       conics = Sigma_v^{-1}
 *       Stored as upper-tri [N, 6]
 *
 * F6. Compute radii (non-differentiable):
 *       For each axis k: radii[k] = ceil(radius_clip * max_scales * r[k])
 *       Stored as int32 [N, 3]
 *
 * =========================================================================
 * Data Layout
 * =========================================================================
 *
 * Input:
 *   means     [N, 3]    -- world-space Gaussian centers (x, y, z)
 *   covars    [N, 6]    -- world-space covariance (upper-tri)
 *   max_scales [N]      -- max activated scale per Gaussian
 *
 * Config:
 *   ranges    [3]       -- physical extent per axis (rx, ry, rz)
 *   centers   [3]       -- physical center per axis (cx, cy, cz)
 *   dim2, dim1, dim0    -- voxel grid dimensions
 *   eps3d               -- blur epsilon (default 1e-4)
 *   radius_clip         -- radius multiplier (default 3.0)
 *
 * Output:
 *   means3d_v [N, 3]    -- voxel-space centers
 *   conics    [N, 6]    -- inverse covariance (upper-tri)
 *   radii     [N, 3]    -- integer radii per axis (int32)
 *
 * Upper-triangular storage for 3x3 symmetric matrix:
 *   [s00, s01, s02, s11, s12, s22]
 *
 * Hessian storage for 6x6 symmetric: 21 elements upper-tri
 *   index(i,j) = i*(11-i)/2 + j,  i <= j
 *
 * =========================================================================
 * Gradient/Hessian variable naming
 * =========================================================================
 *
 *   v_  prefix: first-order gradient (VJP)
 *   h_  prefix: second-order Hessian
 *
 * Author: Dy164
 * Date: 2026-02-21
 */

#ifndef GSPLAT2CT_WORLD_TO_VOLUME_MAPPING_H
#define GSPLAT2CT_WORLD_TO_VOLUME_MAPPING_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

// ============================================================================
// Forward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for world-to-volume mapping forward pass.
 *
 * @tparam S Scalar type (float, double)
 *
 * @param means       World-space Gaussian centers [N, 3]
 * @param covars      World-space covariance upper-tri [N, 6]
 * @param max_scales  Max activated scale per Gaussian [N]
 * @param ranges      Physical extent per axis [3] (rx, ry, rz)
 * @param centers     Physical center per axis [3] (cx, cy, cz)
 * @param dim2        Voxel grid x dimension (fastest)
 * @param dim1        Voxel grid y dimension (middle)
 * @param dim0        Voxel grid z dimension (slowest)
 * @param eps3d       Blur epsilon added to diagonal
 * @param radius_clip Radius multiplier for radii computation
 * @param means3d_v   Output voxel-space centers [N, 3]
 * @param conics      Output inverse covariance upper-tri [N, 6]
 * @param radii       Output integer radii per axis [N, 3] (int32)
 * @param N           Number of Gaussians
 * @param stream      CUDA stream
 */
template <typename S>
void launch_world_to_volume_mapping_fwd(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ max_scales,  // [N]
    const S* __restrict__ ranges,      // [3]
    const S* __restrict__ centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    S eps3d,
    S radius_clip,
    S* __restrict__ means3d_v,         // [N, 3]
    S* __restrict__ conics,            // [N, 6]
    int32_t* __restrict__ radii,       // [N, 3]
    uint32_t N,
    cudaStream_t stream
);

// ============================================================================
// First-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for world-to-volume mapping backward pass (gradients).
 *
 * Computes:
 *   v_means  = dL/d(means)
 *   v_covars = dL/d(covars)
 *
 * from downstream gradients v_means3d_v and v_conics.
 * Opacity gradient is passed through directly (not computed here).
 *
 * @tparam S Scalar type
 *
 * @param means       Forward input [N, 3]
 * @param covars      Forward input [N, 6]
 * @param ranges      Physical extent [3]
 * @param centers     Physical center [3]
 * @param dim2,dim1,dim0 Voxel dimensions
 * @param eps3d       Blur epsilon
 * @param v_means3d_v Downstream dL/d(means3d_v) [N, 3]
 * @param v_conics    Downstream dL/d(conics) [N, 6]
 * @param v_means     Output dL/d(means) [N, 3]
 * @param v_covars    Output dL/d(covars) [N, 6]
 * @param N           Number of Gaussians
 * @param stream      CUDA stream
 */
template <typename S>
void launch_world_to_volume_mapping_bwd(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ ranges,      // [3]
    const S* __restrict__ centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    S eps3d,
    const S* __restrict__ v_means3d_v, // [N, 3]
    const S* __restrict__ v_conics,    // [N, 6]
    S* __restrict__ v_means,           // [N, 3]
    S* __restrict__ v_covars,          // [N, 6]
    uint32_t N,
    cudaStream_t stream
);

// ============================================================================
// Second-Order Backward Pass Launch Function
// ============================================================================

/**
 * Launch kernel for world-to-volume mapping second-order backward pass.
 *
 * Given downstream first-order gradients and second-order Hessians w.r.t.
 * (means3d_v, conics), plus cross-Hessians with opacity, compute upstream
 * gradients and Hessians w.r.t. (means, covars), plus cross-Hessians.
 *
 * The means transform is affine (linear), so its Hessian contribution
 * is zero. All second-order terms come from the matrix inverse in
 * covars → conics.
 *
 * @tparam S Scalar type
 *
 * @param means             Forward input [N, 3]
 * @param covars            Forward input [N, 6]
 * @param ranges            Physical extent [3]
 * @param centers           Physical center [3]
 * @param dim2,dim1,dim0    Voxel dimensions
 * @param eps3d             Blur epsilon
 * @param v_means3d_v       Downstream dL/d(means3d_v) [N, 3]
 * @param v_conics          Downstream dL/d(conics) [N, 6]
 * @param h_means3d_v       Downstream H_{m,m} upper-tri [N, 6]
 * @param h_conics          Downstream H_{c,c} upper-tri [N, 21]
 * @param h_means3d_v_conics Downstream H_{m,c} [N, 18]
 * @param h_means3d_v_opacity Downstream H_{m,op} [N, 3]
 * @param h_conics_opacity  Downstream H_{c,op} [N, 6]
 * @param v_means           Output dL/d(means) [N, 3]
 * @param v_covars          Output dL/d(covars) [N, 6]
 * @param h_means           Output H_{p,p} upper-tri [N, 6]
 * @param h_covars          Output H_{s,s} upper-tri [N, 21]
 * @param h_means_covars    Output H_{p,s} [N, 18]
 * @param h_means_opacity   Output H_{p,op} [N, 3]
 * @param h_covars_opacity  Output H_{s,op} [N, 6]
 * @param N                 Number of Gaussians
 * @param stream            CUDA stream
 */
template <typename S>
void launch_world_to_volume_mapping_bwd2(
    // Forward inputs (for recomputation)
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ ranges,      // [3]
    const S* __restrict__ centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    S eps3d,
    // Downstream first-order gradients
    const S* __restrict__ v_means3d_v, // [N, 3]
    const S* __restrict__ v_conics,    // [N, 6]
    // Downstream phi-phi Hessians
    const S* __restrict__ h_means3d_v,          // [N, 6]  upper-tri 3x3
    const S* __restrict__ h_conics,              // [N, 21] upper-tri 6x6
    const S* __restrict__ h_means3d_v_conics,    // [N, 18] 3x6 row-major
    // Downstream phi-psi cross Hessians (psi = opacity, Dpsi=1)
    const S* __restrict__ h_means3d_v_opacity,   // [N, 3]
    const S* __restrict__ h_conics_opacity,      // [N, 6]
    // Outputs: first-order gradients
    S* __restrict__ v_means,           // [N, 3]
    S* __restrict__ v_covars,          // [N, 6]
    // Outputs: theta-theta Hessians
    S* __restrict__ h_means,           // [N, 6]  upper-tri 3x3
    S* __restrict__ h_covars,          // [N, 21] upper-tri 6x6
    S* __restrict__ h_means_covars,    // [N, 18] 3x6 row-major
    // Outputs: theta-psi cross Hessians
    S* __restrict__ h_means_opacity,   // [N, 3]
    S* __restrict__ h_covars_opacity,  // [N, 6]
    // Dimensions
    uint32_t N,
    cudaStream_t stream
);

} // namespace gsplat2ct

#endif // GSPLAT2CT_WORLD_TO_VOLUME_MAPPING_H
