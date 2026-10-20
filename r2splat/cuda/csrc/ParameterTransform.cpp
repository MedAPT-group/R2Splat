/**
 * Tensor wrappers for (nonunit quaternion, physical scales) -> covariance.
 * Author: Dy164
 */
#include "ParameterTransform.h"
#include "../include/Ops.h"
#include <c10/cuda/CUDAException.h>
#include <limits>

namespace r2splat {
namespace {

void check_parameters(const torch::Tensor& quats, const torch::Tensor& scales) {
    R2SPLAT_CHECK_INPUT(quats);
    R2SPLAT_CHECK_INPUT(scales);
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4,
                "quats must have shape [N, 4], got ", quats.sizes());
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3,
                "scales must have shape [N, 3], got ", scales.sizes());
    TORCH_CHECK(quats.scalar_type() == torch::kFloat32 ||
                quats.scalar_type() == torch::kFloat64,
                "quats must have dtype float32 or float64");
    TORCH_CHECK(scales.scalar_type() == quats.scalar_type(),
                "scales must have the same dtype as quats");
    TORCH_CHECK(scales.device() == quats.device(),
                "scales must be on the same CUDA device as quats");
    TORCH_CHECK(scales.size(0) == quats.size(0),
                "scales and quats must have the same batch size");
    TORCH_CHECK(quats.size(0) <= std::numeric_limits<int32_t>::max(),
                "parameter count exceeds the supported int32 range");
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor>
parameter_transform_fwd_tensor(
    const torch::Tensor& quats, const torch::Tensor& scales
) {
    check_parameters(quats, scales);
    R2SPLAT_DEVICE_GUARD(quats);
    const int64_t N = quats.size(0);
    auto covars = torch::empty({N, 6}, quats.options());
    auto max_scales = torch::empty({N}, quats.options());
    if (N == 0) return std::make_tuple(covars, max_scales);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(quats.scalar_type(), "parameter_transform_fwd", [&]() {
        launch_parameter_transform_fwd<scalar_t>(
            quats.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(),
            covars.data_ptr<scalar_t>(), max_scales.data_ptr<scalar_t>(),
            static_cast<uint32_t>(N), stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(covars, max_scales);
}

std::tuple<torch::Tensor, torch::Tensor>
parameter_transform_bwd_tensor(
    const torch::Tensor& quats, const torch::Tensor& scales,
    const torch::Tensor& v_covars
) {
    check_parameters(quats, scales);
    R2SPLAT_CHECK_INPUT(v_covars);
    TORCH_CHECK(v_covars.dim() == 2 && v_covars.size(1) == 6 &&
                v_covars.size(0) == quats.size(0),
                "v_covars must have shape [N, 6], got ", v_covars.sizes());
    TORCH_CHECK(v_covars.scalar_type() == quats.scalar_type(),
                "v_covars must have the same dtype as quats");
    TORCH_CHECK(v_covars.device() == quats.device(),
                "v_covars must be on the same CUDA device as quats");
    R2SPLAT_DEVICE_GUARD(quats);
    const int64_t N = quats.size(0);
    auto v_quats = torch::empty({N, 4}, quats.options());
    auto v_scales = torch::empty({N, 3}, quats.options());
    if (N == 0) return std::make_tuple(v_quats, v_scales);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(quats.scalar_type(), "parameter_transform_bwd", [&]() {
        launch_parameter_transform_bwd<scalar_t>(
            quats.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(),
            v_covars.data_ptr<scalar_t>(), v_quats.data_ptr<scalar_t>(),
            v_scales.data_ptr<scalar_t>(), static_cast<uint32_t>(N), stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(v_quats, v_scales);
}

} // namespace r2splat
