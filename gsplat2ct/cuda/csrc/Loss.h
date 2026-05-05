/**
 * CUDA kernel launch declarations for image and volume losses.
 */

#ifndef GSPLAT2CT_LOSS_H
#define GSPLAT2CT_LOSS_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace gsplat2ct {

template <typename S>
void launch_image_loss_l2_fwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    S* __restrict__ l2_map,
    S* __restrict__ mu_r,
    S* __restrict__ mu_m,
    S* __restrict__ f1,
    S* __restrict__ f2,
    S* __restrict__ f3,
    S* __restrict__ f4,
    S* __restrict__ ssim_map,
    int B, int H, int W,
    int window_size,
    bool compute_statistics,
    cudaStream_t stream);

template <typename S>
void launch_image_loss_l2_bwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    const S* __restrict__ mu_r,
    const S* __restrict__ mu_m,
    const S* __restrict__ f1,
    const S* __restrict__ f2,
    const S* __restrict__ f3,
    const S* __restrict__ f4,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    int B, int H, int W,
    int window_size,
    S lambda_l2,
    S lambda_ssim,
    S C1, S C2,
    cudaStream_t stream);

template <typename S>
void launch_image_loss_l1_fwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    S* __restrict__ l1_map,
    S* __restrict__ mu_r,
    S* __restrict__ mu_m,
    S* __restrict__ f1,
    S* __restrict__ f2,
    S* __restrict__ f3,
    S* __restrict__ f4,
    S* __restrict__ ssim_map,
    int B, int H, int W,
    int window_size,
    bool compute_statistics,
    cudaStream_t stream);

template <typename S>
void launch_image_loss_l1_bwd(
    const S* __restrict__ prediction,
    const S* __restrict__ target,
    const S* __restrict__ gaussian_window,
    const S* __restrict__ mu_r,
    const S* __restrict__ mu_m,
    const S* __restrict__ f1,
    const S* __restrict__ f2,
    const S* __restrict__ f3,
    const S* __restrict__ f4,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_pred,
    int B, int H, int W,
    int window_size,
    S lambda_l1,
    S lambda_ssim,
    S C1, S C2,
    cudaStream_t stream);

template <typename S>
void launch_volume_loss_fwd(
    const S* __restrict__ volume,
    S* __restrict__ tv_loss,
    S* __restrict__ tv_map,
    int B, int D, int H, int W,
    S eps,
    cudaStream_t stream);

template <typename S>
void launch_volume_loss_bwd(
    const S* __restrict__ volume,
    const S* __restrict__ grad_output,
    S* __restrict__ grad_volume,
    int B, int D, int H, int W,
    S eps,
    cudaStream_t stream);

} // namespace gsplat2ct

#endif // GSPLAT2CT_LOSS_H
