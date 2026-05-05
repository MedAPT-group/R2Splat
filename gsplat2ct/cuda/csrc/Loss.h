/**
 * CUDA kernel launch declarations for image and volume losses.
 */

#ifndef GSPLAT2CT_LOSS_H
#define GSPLAT2CT_LOSS_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

template <typename S>
void launch_image_loss_l1_fwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    S* __restrict__ l1_map,
    int B,
    int H,
    int W,
    cudaStream_t stream);

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
    cudaStream_t stream);

template <typename S>
void launch_volume_loss_fwd(
    const S* __restrict__ volume,
    S* __restrict__ tv_loss,
    S* __restrict__ tv_map,
    int B,
    int D,
    int H,
    int W,
    cudaStream_t stream);

template <typename S>
void launch_volume_loss_bwd(
    const S* __restrict__ volume,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_volume,
    int B,
    int D,
    int H,
    int W,
    cudaStream_t stream);

} // namespace gsplat2ct

#endif // GSPLAT2CT_LOSS_H
