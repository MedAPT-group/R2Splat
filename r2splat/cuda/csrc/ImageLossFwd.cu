/**
 * ImageLoss Forward CUDA Kernels.
 *
 * Training uses only mean L1 image loss.
 */

#include "Loss.h"
#include "../include/common.h"
#include <cuda_runtime.h>
#include <cmath>

namespace r2splat {

template <typename S>
__device__ __forceinline__ S abs_func(S x) {
    return x >= static_cast<S>(0) ? x : -x;
}

template <typename S>
__global__ void compute_l1_map_kernel(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    S* __restrict__ l1_map,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    S diff = prediction[idx] - target[idx];
    l1_map[idx] = abs_func(diff);
}

template <typename S>
void launch_image_loss_l1_fwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    S* __restrict__ l1_map,
    int B,
    int H,
    int W,
    cudaStream_t stream
) {
    int total_pixels = B * H * W;
    int threads = R2SPLAT_N_THREADS;
    int blocks = (total_pixels + threads - 1) / threads;

    compute_l1_map_kernel<S><<<blocks, threads, 0, stream>>>(
        prediction, target, l1_map, total_pixels
    );
}

template void launch_image_loss_l1_fwd<float>(
    const float*, const float*, float*, int, int, int, cudaStream_t
);

template void launch_image_loss_l1_fwd<double>(
    const double*, const double*, double*, int, int, int, cudaStream_t
);

} // namespace r2splat
