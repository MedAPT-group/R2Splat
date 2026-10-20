/**
 * First-order VJP through covariance and quaternion normalization.
 * Author: Dy164
 */
#include "ParameterTransform.h"
#include "../include/ParameterTransform.cuh"

namespace r2splat {

template <typename S>
__global__ void parameter_transform_bwd_kernel(
    const S* __restrict__ quats, const S* __restrict__ scales,
    const S* __restrict__ v_covars, S* __restrict__ v_quats,
    S* __restrict__ v_scales, uint32_t N
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const size_t offset = idx;

    S q[4], magnitude, scaled_norm;
    normalize_parameter_quaternion(quats + offset * 4, q, magnitude, scaled_norm);
    const S w = q[0], x = q[1], y = q[2], z = q[3];
    S R[3][3];
    parameter_rotation(q, R);
    const S* s = scales + offset * 3;
    const S* vc = v_covars + offset * 6;

    // A stored off-diagonal occurs only once: split its VJP equally between
    // the two entries of the full symmetric covariance gradient.
    const S G[3][3] = {
        {vc[0], S(0.5) * vc[1], S(0.5) * vc[2]},
        {S(0.5) * vc[1], vc[3], S(0.5) * vc[4]},
        {S(0.5) * vc[2], S(0.5) * vc[4], vc[5]}
    };
    S vR[3][3];
    S* vs = v_scales + offset * 3;
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        S scale_gradient = 0;
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            // M = R diag(s), vM = 2 G M; vR_ij = vM_ij * s_j.
            S vM = 0;
            #pragma unroll
            for (int k = 0; k < 3; ++k) vM += G[i][k] * R[k][j];
            vM *= S(2) * s[j];
            vR[i][j] = vM * s[j];
            scale_gradient += R[i][j] * vM;
        }
        vs[j] = scale_gradient;
    }

    const S g[4] = {
        S(2) * (x * (vR[2][1] - vR[1][2]) +
                y * (vR[0][2] - vR[2][0]) + z * (vR[1][0] - vR[0][1])),
        S(2) * (-S(2) * x * (vR[1][1] + vR[2][2]) +
                y * (vR[0][1] + vR[1][0]) + z * (vR[0][2] + vR[2][0]) +
                w * (vR[2][1] - vR[1][2])),
        S(2) * (x * (vR[0][1] + vR[1][0]) -
                S(2) * y * (vR[0][0] + vR[2][2]) +
                z * (vR[1][2] + vR[2][1]) + w * (vR[0][2] - vR[2][0])),
        S(2) * (x * (vR[0][2] + vR[2][0]) + y * (vR[1][2] + vR[2][1]) -
                S(2) * z * (vR[0][0] + vR[1][1]) + w * (vR[1][0] - vR[0][1]))
    };
    S q_dot_g = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) q_dot_g += q[i] * g[i];

    S* vq = v_quats + offset * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        // d normalize(q)^T g = (g - q_hat (q_hat^T g)) / ||q||.
        // Divide sequentially to avoid overflowing/underflowing the raw norm.
        vq[i] = parameter_divide(
            parameter_divide(g[i] - q[i] * q_dot_g, scaled_norm), magnitude);
    }
}

template <typename S>
void launch_parameter_transform_bwd(
    const S* __restrict__ quats, const S* __restrict__ scales,
    const S* __restrict__ v_covars, S* __restrict__ v_quats,
    S* __restrict__ v_scales, uint32_t N, cudaStream_t stream
) {
    if (N == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (N - 1) / threads + 1;
    parameter_transform_bwd_kernel<S><<<blocks, threads, 0, stream>>>(
        quats, scales, v_covars, v_quats, v_scales, N);
}

template void launch_parameter_transform_bwd<float>(
    const float*, const float*, const float*, float*, float*, uint32_t, cudaStream_t);
template void launch_parameter_transform_bwd<double>(
    const double*, const double*, const double*, double*, double*, uint32_t, cudaStream_t);

} // namespace r2splat
