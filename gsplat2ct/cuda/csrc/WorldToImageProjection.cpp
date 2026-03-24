/**
 * @file WorldToImageProjection.cpp
 * @brief C++ Tensor Wrapper for World-to-Image Projection Module
 *
 * Provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 *
 * Input shapes:
 *   means:     [N, 3]    -- world-space Gaussian centers (shared across cameras)
 *   covars:    [N, 6]    -- world-space covariance upper-tri (shared across cameras)
 *   viewmats:  [C, 4, 4] -- camera extrinsic matrices
 *   Ks:        [C, 3, 3] -- camera intrinsic matrices
 *
 * Output shapes:
 *   means2d:   [C, N, 2]  -- projected 2D centers
 *   cov2d_3x3: [C, N, 6]  -- projected 3x3 covariance (upper-tri)
 *
 * Following gsplat2ct's pattern for tensor handling.
 *
 * Author: Dy164
 * Date: 2026-02-19
 */

#include "WorldToImageProjection.h"
#include "../include/Ops.h"
#include <torch/extension.h>

namespace gsplat2ct {

// ============================================================================
// Forward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_image_projection_fwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& viewmats,    // [C, 4, 4]
    const torch::Tensor& Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    float eps_blur
) {
    GSPLAT2CT_DEVICE_GUARD(means);
    GSPLAT2CT_CHECK_INPUT(means);
    GSPLAT2CT_CHECK_INPUT(covars);
    GSPLAT2CT_CHECK_INPUT(viewmats);
    GSPLAT2CT_CHECK_INPUT(Ks);

    // Validate dimensions
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
                "means must be 2D [N, 3], got shape ",
                means.sizes());
    TORCH_CHECK(covars.dim() == 2 && covars.size(1) == 6,
                "covars must be 2D [N, 6], got shape ",
                covars.sizes());
    TORCH_CHECK(viewmats.dim() == 3 && viewmats.size(1) == 4 && viewmats.size(2) == 4,
                "viewmats must be 3D [C, 4, 4], got shape ",
                viewmats.sizes());
    TORCH_CHECK(Ks.dim() == 3 && Ks.size(1) == 3 && Ks.size(2) == 3,
                "Ks must be 3D [C, 3, 3], got shape ",
                Ks.sizes());

    const int64_t N = means.size(0);
    const int64_t C = viewmats.size(0);
    const uint32_t total = static_cast<uint32_t>(C * N);

    // Validate shape consistency
    TORCH_CHECK(covars.size(0) == N,
                "covars dim 0 (", covars.size(0),
                ") doesn't match N=", N);
    TORCH_CHECK(Ks.size(0) == C,
                "Ks batch dim (", Ks.size(0),
                ") doesn't match C=", C);

    // Allocate outputs
    torch::Tensor means2d = torch::empty(
        {C, N, 2}, means.options());
    torch::Tensor cov2d_3x3 = torch::empty(
        {C, N, 6}, means.options());
    torch::Tensor radii = torch::zeros(
        {C, N}, means.options().dtype(torch::kInt32));

    if (total == 0) {
        return std::make_tuple(means2d, cov2d_3x3, radii);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "world_to_image_projection_fwd",
        [&]() {
            launch_world_to_image_projection_fwd<scalar_t>(
                means.data_ptr<scalar_t>(),
                covars.data_ptr<scalar_t>(),
                viewmats.data_ptr<scalar_t>(),
                Ks.data_ptr<scalar_t>(),
                dim1,
                dim0,
                static_cast<scalar_t>(eps_blur),
                means2d.data_ptr<scalar_t>(),
                cov2d_3x3.data_ptr<scalar_t>(),
                radii.data_ptr<int32_t>(),
                static_cast<uint32_t>(C),
                static_cast<uint32_t>(N),
                stream
            );
        }
    );

    return std::make_tuple(means2d, cov2d_3x3, radii);
}

// ============================================================================
// First-Order Backward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
world_to_image_projection_bwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& viewmats,    // [C, 4, 4]
    const torch::Tensor& Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    const torch::Tensor& v_means2d,   // [C, N, 2]
    const torch::Tensor& v_cov2d_3x3  // [C, N, 6]
) {
    GSPLAT2CT_DEVICE_GUARD(means);
    GSPLAT2CT_CHECK_INPUT(means);
    GSPLAT2CT_CHECK_INPUT(covars);
    GSPLAT2CT_CHECK_INPUT(viewmats);
    GSPLAT2CT_CHECK_INPUT(Ks);
    GSPLAT2CT_CHECK_INPUT(v_means2d);
    GSPLAT2CT_CHECK_INPUT(v_cov2d_3x3);

    // Validate dimensions
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
                "means must be 2D [N, 3]");
    TORCH_CHECK(covars.dim() == 2 && covars.size(1) == 6,
                "covars must be 2D [N, 6]");
    TORCH_CHECK(viewmats.dim() == 3 && viewmats.size(1) == 4 && viewmats.size(2) == 4,
                "viewmats must be 3D [C, 4, 4]");
    TORCH_CHECK(Ks.dim() == 3 && Ks.size(1) == 3 && Ks.size(2) == 3,
                "Ks must be 3D [C, 3, 3]");
    TORCH_CHECK(v_means2d.dim() == 3 && v_means2d.size(2) == 2,
                "v_means2d must be 3D [C, N, 2]");
    TORCH_CHECK(v_cov2d_3x3.dim() == 3 && v_cov2d_3x3.size(2) == 6,
                "v_cov2d_3x3 must be 3D [C, N, 6]");

    const int64_t N = means.size(0);
    const int64_t C = viewmats.size(0);
    const uint32_t total = static_cast<uint32_t>(C * N);

    // Validate shape consistency
    TORCH_CHECK(covars.size(0) == N,
                "covars dim 0 mismatch");
    TORCH_CHECK(v_means2d.size(0) == C && v_means2d.size(1) == N,
                "v_means2d shape mismatch");
    TORCH_CHECK(v_cov2d_3x3.size(0) == C && v_cov2d_3x3.size(1) == N,
                "v_cov2d_3x3 shape mismatch");

    // Allocate outputs: accumulated over C cameras via atomicAdd in kernel
    torch::Tensor v_means = torch::empty(
        {N, 3}, means.options());
    torch::Tensor v_covars = torch::empty(
        {N, 6}, means.options());

    if (total == 0) {
        return std::make_tuple(v_means, v_covars);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "world_to_image_projection_bwd",
        [&]() {
            launch_world_to_image_projection_bwd<scalar_t>(
                means.data_ptr<scalar_t>(),
                covars.data_ptr<scalar_t>(),
                viewmats.data_ptr<scalar_t>(),
                Ks.data_ptr<scalar_t>(),
                dim1,
                dim0,
                v_means2d.data_ptr<scalar_t>(),
                v_cov2d_3x3.data_ptr<scalar_t>(),
                v_means.data_ptr<scalar_t>(),
                v_covars.data_ptr<scalar_t>(),
                static_cast<uint32_t>(C),
                static_cast<uint32_t>(N),
                stream
            );
        }
    );

    return std::make_tuple(v_means, v_covars);
}

// ============================================================================
// Second-Order Backward Pass Tensor Interface
// ============================================================================

std::tuple<
    torch::Tensor, torch::Tensor,   // v_means, v_covars
    torch::Tensor, torch::Tensor, torch::Tensor,  // h_means, h_covars, h_means_covars
    torch::Tensor, torch::Tensor    // h_means_opacity, h_covars_opacity
>
world_to_image_projection_bwd2_tensor(
    const torch::Tensor& means,                // [N, 3]
    const torch::Tensor& covars,               // [N, 6]
    const torch::Tensor& viewmats,             // [C, 4, 4]
    const torch::Tensor& Ks,                   // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    // Downstream first-order gradients
    const torch::Tensor& v_means2d,            // [C, N, 2]
    const torch::Tensor& v_cov2d_3x3,         // [C, N, 6]
    // Downstream phi-phi Hessians
    const torch::Tensor& h_means2d,            // [C, N, 3]
    const torch::Tensor& h_cov2d_3x3,         // [C, N, 21]
    const torch::Tensor& h_means2d_cov2d_3x3, // [C, N, 12]
    // Downstream phi-psi cross Hessians
    const torch::Tensor& h_means2d_opacity,    // [C, N, 2]
    const torch::Tensor& h_cov2d_3x3_opacity  // [C, N, 6]
) {
    GSPLAT2CT_DEVICE_GUARD(means);
    GSPLAT2CT_CHECK_INPUT(means);
    GSPLAT2CT_CHECK_INPUT(covars);
    GSPLAT2CT_CHECK_INPUT(viewmats);
    GSPLAT2CT_CHECK_INPUT(Ks);
    GSPLAT2CT_CHECK_INPUT(v_means2d);
    GSPLAT2CT_CHECK_INPUT(v_cov2d_3x3);
    GSPLAT2CT_CHECK_INPUT(h_means2d);
    GSPLAT2CT_CHECK_INPUT(h_cov2d_3x3);
    GSPLAT2CT_CHECK_INPUT(h_means2d_cov2d_3x3);
    GSPLAT2CT_CHECK_INPUT(h_means2d_opacity);
    GSPLAT2CT_CHECK_INPUT(h_cov2d_3x3_opacity);

    // Validate dimensions
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
                "means must be 2D [N, 3]");
    TORCH_CHECK(covars.dim() == 2 && covars.size(1) == 6,
                "covars must be 2D [N, 6]");
    TORCH_CHECK(viewmats.dim() == 3 && viewmats.size(1) == 4 && viewmats.size(2) == 4,
                "viewmats must be 3D [C, 4, 4]");
    TORCH_CHECK(Ks.dim() == 3 && Ks.size(1) == 3 && Ks.size(2) == 3,
                "Ks must be 3D [C, 3, 3]");
    TORCH_CHECK(v_means2d.dim() == 3 && v_means2d.size(2) == 2,
                "v_means2d must be 3D [C, N, 2]");
    TORCH_CHECK(v_cov2d_3x3.dim() == 3 && v_cov2d_3x3.size(2) == 6,
                "v_cov2d_3x3 must be 3D [C, N, 6]");
    TORCH_CHECK(h_means2d.dim() == 3 && h_means2d.size(2) == 3,
                "h_means2d must be 3D [C, N, 3]");
    TORCH_CHECK(h_cov2d_3x3.dim() == 3 && h_cov2d_3x3.size(2) == 21,
                "h_cov2d_3x3 must be 3D [C, N, 21]");
    TORCH_CHECK(h_means2d_cov2d_3x3.dim() == 3 && h_means2d_cov2d_3x3.size(2) == 12,
                "h_means2d_cov2d_3x3 must be 3D [C, N, 12]");
    TORCH_CHECK(h_means2d_opacity.dim() == 3 && h_means2d_opacity.size(2) == 2,
                "h_means2d_opacity must be 3D [C, N, 2]");
    TORCH_CHECK(h_cov2d_3x3_opacity.dim() == 3 && h_cov2d_3x3_opacity.size(2) == 6,
                "h_cov2d_3x3_opacity must be 3D [C, N, 6]");

    const int64_t N = means.size(0);
    const int64_t C = viewmats.size(0);
    const uint32_t total = static_cast<uint32_t>(C * N);

    // Validate shape consistency
    TORCH_CHECK(covars.size(0) == N,
                "covars dim 0 mismatch");
    TORCH_CHECK(v_means2d.size(0) == C && v_means2d.size(1) == N,
                "v_means2d shape mismatch");
    TORCH_CHECK(v_cov2d_3x3.size(0) == C && v_cov2d_3x3.size(1) == N,
                "v_cov2d_3x3 shape mismatch");
    TORCH_CHECK(h_means2d.size(0) == C && h_means2d.size(1) == N,
                "h_means2d shape mismatch");
    TORCH_CHECK(h_cov2d_3x3.size(0) == C && h_cov2d_3x3.size(1) == N,
                "h_cov2d_3x3 shape mismatch");
    TORCH_CHECK(h_means2d_cov2d_3x3.size(0) == C && h_means2d_cov2d_3x3.size(1) == N,
                "h_means2d_cov2d_3x3 shape mismatch");

    // Allocate outputs
    // First-order gradients (accumulated over C cameras)
    torch::Tensor v_means = torch::empty(
        {N, 3}, means.options());
    torch::Tensor v_covars = torch::empty(
        {N, 6}, means.options());

    // Theta-theta Hessians (accumulated over C cameras)
    torch::Tensor h_means_out = torch::empty(
        {N, 6}, means.options());        // upper-tri 3x3
    torch::Tensor h_covars_out = torch::empty(
        {N, 21}, means.options());       // upper-tri 6x6
    torch::Tensor h_means_covars = torch::empty(
        {N, 18}, means.options());       // 3x6 row-major

    // Theta-psi cross Hessians (per-camera, opacity is per-camera)
    torch::Tensor h_means_opacity = torch::empty(
        {C, N, 3}, means.options());
    torch::Tensor h_covars_opacity = torch::empty(
        {C, N, 6}, means.options());

    if (total == 0) {
        return std::make_tuple(
            v_means, v_covars,
            h_means_out, h_covars_out, h_means_covars,
            h_means_opacity, h_covars_opacity
        );
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "world_to_image_projection_bwd2",
        [&]() {
            launch_world_to_image_projection_bwd2<scalar_t>(
                means.data_ptr<scalar_t>(),
                covars.data_ptr<scalar_t>(),
                viewmats.data_ptr<scalar_t>(),
                Ks.data_ptr<scalar_t>(),
                dim1,
                dim0,
                v_means2d.data_ptr<scalar_t>(),
                v_cov2d_3x3.data_ptr<scalar_t>(),
                h_means2d.data_ptr<scalar_t>(),
                h_cov2d_3x3.data_ptr<scalar_t>(),
                h_means2d_cov2d_3x3.data_ptr<scalar_t>(),
                h_means2d_opacity.data_ptr<scalar_t>(),
                h_cov2d_3x3_opacity.data_ptr<scalar_t>(),
                v_means.data_ptr<scalar_t>(),
                v_covars.data_ptr<scalar_t>(),
                h_means_out.data_ptr<scalar_t>(),
                h_covars_out.data_ptr<scalar_t>(),
                h_means_covars.data_ptr<scalar_t>(),
                h_means_opacity.data_ptr<scalar_t>(),
                h_covars_opacity.data_ptr<scalar_t>(),
                static_cast<uint32_t>(C),
                static_cast<uint32_t>(N),
                stream
            );
        }
    );

    return std::make_tuple(
        v_means, v_covars,
        h_means_out, h_covars_out, h_means_covars,
        h_means_opacity, h_covars_opacity
    );
}

} // namespace gsplat2ct
