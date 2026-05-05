/**
 * Loss function C++ tensor wrappers.
 */

#include "Loss.h"
#include "../include/Ops.h"
#include <algorithm>
#include <torch/extension.h>

namespace gsplat2ct {

ImageLossResult image_loss_l1_fwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float lambda_l1
) {
    GSPLAT2CT_CHECK_INPUT(prediction);
    GSPLAT2CT_CHECK_INPUT(target);
    TORCH_CHECK(prediction.dim() == 3, "prediction must be 3D (B, H, W)");
    TORCH_CHECK(target.dim() == 3, "target must be 3D (B, H, W)");
    TORCH_CHECK(
        prediction.sizes() == target.sizes(),
        "prediction and target must have same shape"
    );

    int B = prediction.size(0);
    int H = prediction.size(1);
    int W = prediction.size(2);

    auto l1_map = torch::empty({B, H, W}, prediction.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(prediction.scalar_type(), "image_loss_l1_fwd", [&] {
        launch_image_loss_l1_fwd<scalar_t>(
            prediction.data_ptr<scalar_t>(),
            target.data_ptr<scalar_t>(),
            l1_map.data_ptr<scalar_t>(),
            B,
            H,
            W,
            stream
        );
    });

    auto l1_loss = l1_map.mean();
    auto loss = lambda_l1 * l1_loss;
    return std::make_tuple(loss, l1_loss, l1_map);
}

torch::Tensor image_loss_l1_bwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& grad_output,
    float lambda_l1
) {
    GSPLAT2CT_CHECK_INPUT(prediction);
    GSPLAT2CT_CHECK_INPUT(target);
    GSPLAT2CT_CHECK_INPUT(grad_output);
    TORCH_CHECK(prediction.dim() == 3, "prediction must be 3D (B, H, W)");
    TORCH_CHECK(target.dim() == 3, "target must be 3D (B, H, W)");
    TORCH_CHECK(
        prediction.sizes() == target.sizes(),
        "prediction and target must have same shape"
    );

    int B = prediction.size(0);
    int H = prediction.size(1);
    int W = prediction.size(2);

    auto grad_pred = torch::empty({B, H, W}, prediction.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(prediction.scalar_type(), "image_loss_l1_bwd", [&] {
        launch_image_loss_l1_bwd<scalar_t>(
            prediction.data_ptr<scalar_t>(),
            target.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_pred.data_ptr<scalar_t>(),
            B,
            H,
            W,
            static_cast<scalar_t>(lambda_l1),
            stream
        );
    });

    return grad_pred;
}

ImageLossResult image_loss_fwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    float lambda_l1
) {
    return image_loss_l1_fwd(prediction, target, lambda_l1);
}

torch::Tensor image_loss_bwd(
    const torch::Tensor& prediction,
    const torch::Tensor& target,
    const torch::Tensor& grad_output,
    float lambda_l1
) {
    return image_loss_l1_bwd(prediction, target, grad_output, lambda_l1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> volume_loss_fwd(
    const torch::Tensor& volume,
    float lambda_tv
) {
    GSPLAT2CT_CHECK_INPUT(volume);
    TORCH_CHECK(volume.dim() == 4, "volume must be 4D (B, D, H, W)");

    int B = volume.size(0);
    int D = volume.size(1);
    int H = volume.size(2);
    int W = volume.size(3);

    auto tv_map = torch::empty({B, D, H, W}, volume.options());
    auto tv_loss = torch::zeros({1}, volume.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(volume.scalar_type(), "volume_loss_fwd", [&] {
        launch_volume_loss_fwd<scalar_t>(
            volume.data_ptr<scalar_t>(),
            tv_loss.data_ptr<scalar_t>(),
            tv_map.data_ptr<scalar_t>(),
            B,
            D,
            H,
            W,
            stream
        );
    });

    int edge_count =
        B * ((D - 1) * H * W + D * (H - 1) * W + D * H * (W - 1));
    edge_count = std::max(edge_count, 1);
    tv_loss = tv_loss / static_cast<float>(edge_count);

    auto loss = lambda_tv * tv_loss;
    return std::make_tuple(loss.squeeze(), tv_loss.squeeze(), tv_map);
}

torch::Tensor volume_loss_bwd(
    const torch::Tensor& volume,
    const torch::Tensor& grad_output,
    float lambda_tv
) {
    GSPLAT2CT_CHECK_INPUT(volume);
    GSPLAT2CT_CHECK_INPUT(grad_output);
    TORCH_CHECK(volume.dim() == 4, "volume must be 4D (B, D, H, W)");

    int B = volume.size(0);
    int D = volume.size(1);
    int H = volume.size(2);
    int W = volume.size(3);

    auto grad_volume = torch::empty({B, D, H, W}, volume.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(volume.scalar_type(), "volume_loss_bwd", [&] {
        launch_volume_loss_bwd<scalar_t>(
            volume.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_volume.data_ptr<scalar_t>(),
            B,
            D,
            H,
            W,
            stream
        );
    });

    return grad_volume * lambda_tv;
}

} // namespace gsplat2ct
