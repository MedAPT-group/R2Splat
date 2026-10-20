/** CUDA launch declarations for physical covariance parameters. */
#ifndef R2SPLAT_PARAMETER_TRANSFORM_H
#define R2SPLAT_PARAMETER_TRANSFORM_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

template <typename S>
void launch_parameter_transform_fwd(
    const S* __restrict__ quats,
    const S* __restrict__ scales,
    S* __restrict__ covars,
    S* __restrict__ max_scales,
    uint32_t N,
    cudaStream_t stream);

template <typename S>
void launch_parameter_transform_bwd(
    const S* __restrict__ quats,
    const S* __restrict__ scales,
    const S* __restrict__ v_covars,
    S* __restrict__ v_quats,
    S* __restrict__ v_scales,
    uint32_t N,
    cudaStream_t stream);

} // namespace r2splat
#endif // R2SPLAT_PARAMETER_TRANSFORM_H
