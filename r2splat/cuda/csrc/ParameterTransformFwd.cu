/**
 * Physical covariance forward: q_hat = q / ||q||, C = R diag(scales^2) R^T.
 * Author: Dy164
 */
#include "ParameterTransform.h"
#include "../include/ParameterTransform.cuh"

namespace r2splat {

template <typename S>
__global__ void parameter_transform_fwd_kernel(
    const S* __restrict__ quats, const S* __restrict__ scales,
    S* __restrict__ covars, S* __restrict__ max_scales, uint32_t N
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const size_t offset = idx;

    S q[4], magnitude, scaled_norm;
    normalize_parameter_quaternion(quats + offset * 4, q, magnitude, scaled_norm);
    S R[3][3];
    parameter_rotation(q, R);
    const S* s = scales + offset * 3;
    max_scales[idx] = max(s[0], max(s[1], s[2]));

    S* out = covars + offset * 6;
    int packed = 0;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = i; j < 3; ++j) {
            S value = 0;
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                value += R[i][k] * R[j][k] * s[k] * s[k];
            }
            out[packed++] = value;
        }
    }
}

template <typename S>
void launch_parameter_transform_fwd(
    const S* __restrict__ quats, const S* __restrict__ scales,
    S* __restrict__ covars, S* __restrict__ max_scales,
    uint32_t N, cudaStream_t stream
) {
    if (N == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (N - 1) / threads + 1;
    parameter_transform_fwd_kernel<S><<<blocks, threads, 0, stream>>>(
        quats, scales, covars, max_scales, N);
}

template void launch_parameter_transform_fwd<float>(
    const float*, const float*, float*, float*, uint32_t, cudaStream_t);
template void launch_parameter_transform_fwd<double>(
    const double*, const double*, double*, double*, uint32_t, cudaStream_t);

} // namespace r2splat
