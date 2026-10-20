/**
 * ImageLoss Backward CUDA Kernels.
 *
 * Training uses only mean L1 image loss.
 */

#include "Loss.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

template <typename S>
__global__ void compute_l1_grad_kernel(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    S lambda_l1,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    S diff = prediction[idx] - target[idx];
    S sign = (diff > static_cast<S>(0))
        ? static_cast<S>(1)
        : ((diff < static_cast<S>(0)) ? static_cast<S>(-1) : static_cast<S>(0));
    grad_pred[idx] = grad_output[0] * lambda_l1 * sign / static_cast<S>(total_elements);
}

template <typename S>
void launch_image_loss_l1_bwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    int B,
    int H,
    int W,
    S lambda_l1,
    cudaStream_t stream
) {
    int total_pixels = B * H * W;
    int threads = R2SPLAT_N_THREADS;
    int blocks = (total_pixels + threads - 1) / threads;

    compute_l1_grad_kernel<S><<<blocks, threads, 0, stream>>>(
        prediction, target, grad_output, grad_pred, lambda_l1, total_pixels
    );
}

template void launch_image_loss_l1_bwd<float>(
    const float*, const float*, const float*, float*,
    int, int, int, float, cudaStream_t
);

template void launch_image_loss_l1_bwd<double>(
    const double*, const double*, const double*, double*,
    int, int, int, double, cudaStream_t
);

} // namespace r2splat
