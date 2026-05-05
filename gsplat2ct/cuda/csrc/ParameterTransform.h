/**
 * CUDA kernel launch declarations for raw Gaussian parameter transforms.
 */

#ifndef GSPLAT2CT_PARAMETER_TRANSFORM_H
#define GSPLAT2CT_PARAMETER_TRANSFORM_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

template <typename S>
void launch_parameter_transform_fwd(
    const S* __restrict__ quats,
    const S* __restrict__ scale_ori,
    S scale_min,
    S scale_max,
    S* __restrict__ covars,
    S* __restrict__ max_scales,
    uint32_t N,
    cudaStream_t stream);

template <typename S>
void launch_parameter_transform_bwd(
    const S* __restrict__ quats,
    const S* __restrict__ scale_ori,
    S scale_min,
    S scale_max,
    const S* __restrict__ v_covars,
    S* __restrict__ v_quats,
    S* __restrict__ v_scale_ori,
    uint32_t N,
    cudaStream_t stream);

} // namespace gsplat2ct

#endif // GSPLAT2CT_PARAMETER_TRANSFORM_H
