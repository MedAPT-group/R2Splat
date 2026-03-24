/*NOTES by Dy164:
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
If test this backward module, errors may arise, especially in conics gradients,
this is firstly because the amplitude of conics is large,
and therefore the absolute error in conics gradients may be large too;
the second reason is that the intersection records are generated using 3 sigma cutoff,
as a consequence of which, cutoff effects at tile boundaries may lead to inaccurate gradients.
To mitigate the cutoff effects, we can increase the cutoff to 4 sigma,
yet this will also increase the memory consumption and computation time.
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
*/
/**
 * RasterizeToPixels Backward CUDA Kernels
 * 
 * This file implements CUDA kernels for tile-based rasterization backward pass
 * specifically designed for CT reconstruction.
 * 
 * Key differences from standard 3DGS rasterization backward:
 * - No alpha blending: CT uses simple summation, no cumulative transmittance
 * - Order-independent: No need to process Gaussians back-to-front
 * - Single channel output: No COLOR_DIM template parameter
 * - No masks support
 * 
 * CT forward formula: pixel_value = Σ (opacity × mu × exp(-sigma))
 * 
 * Gradients computed:
 * - v_means2d: gradient w.r.t. 2D centers
 * - v_conics: gradient w.r.t. inverse covariance matrices
 * - v_opacities: gradient w.r.t. opacities
 * - v_mus: gradient w.r.t. mu values
 * 
 * Author: Dy164
 * Date: 2025-01-27
 */

#include "RasterizeToPixels.h"
#include "../include/types.cuh"
#include "../include/helpers.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace gsplat2ct {

namespace cg = cooperative_groups;

// ============================================================================
// Backward Pass Kernel
// ============================================================================

/**
 * Backward pass for CT rasterization.
 * 
 * Each block handles one tile (tile_size × tile_size pixels).
 * Threads cooperatively load Gaussian data and accumulate gradients.
 * 
 * For CT reconstruction (simple summation, no alpha blending):
 *   O = Σ_i (opacity_i × mu_i × exp(-sigma_i))
 *   
 * Given v_render = ∂L/∂O from upstream, we compute:
 *   v_opacity_i = v_render × mu_i × exp(-sigma_i)
 *   v_mu_i = v_render × opacity_i × exp(-sigma_i)
 *   v_sigma_i = -v_render × opacity_i × mu_i × exp(-sigma_i) = -v_render × alpha_i
 *   
 * Then expand sigma = 0.5 × (a×δx² + c×δy²) + b×δx×δy where δ = (px - mean_x, py - mean_y):
 *   v_mean_x = v_sigma × (a×δx + b×δy) (note: ∂sigma/∂mean_x = -∂sigma/∂δx × 1)
 *   v_mean_y = v_sigma × (c×δy + b×δx)
 *   v_conic_a = v_sigma × 0.5 × δx²
 *   v_conic_b = v_sigma × δx × δy
 *   v_conic_c = v_sigma × 0.5 × δy²
 */
template <typename S>
__global__ void rasterize_to_pixels_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    // Forward inputs
    const vec2<S>* __restrict__ means2d,    // [C * N, 2]
    const vec3<S>* __restrict__ conics,     // [C * N, 3]
    const S* __restrict__ opacities,        // [C * N]
    const S* __restrict__ mus,              // [C * N]
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0,
    const int32_t* __restrict__ tile_offsets,  // [C, n_tiles_dim0, n_tiles_dim1]
    const int32_t* __restrict__ flatten_ids,   // [n_isects]
    // Upstream gradient
    const S* __restrict__ v_render_output,     // [C, dim0, dim1]
    // Output gradients
    vec2<S>* __restrict__ v_means2d,           // [C * N, 2]
    vec3<S>* __restrict__ v_conics,            // [C * N, 3]
    S* __restrict__ v_opacities,               // [C * N]
    S* __restrict__ v_mus                      // [C * N]
) {
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * n_tiles_dim1 + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // Offset pointers for this camera
    tile_offsets += camera_id * n_tiles_dim0 * n_tiles_dim1;
    v_render_output += camera_id * dim0 * dim1;

    // Pixel center coordinates
    const S px = static_cast<S>(j) + static_cast<S>(0.5);
    const S py = static_cast<S>(i) + static_cast<S>(0.5);
    const int32_t pix_id = i * dim1 + j;

    // Check if inside valid pixel range
    const bool inside = (i < dim0 && j < dim1);

    // Get upstream gradient for this pixel
    const S v_render = inside ? v_render_output[pix_id] : static_cast<S>(0);

    // Which Gaussians to look through in this tile
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end = 
        (camera_id == static_cast<int32_t>(C) - 1) && 
        (tile_id == static_cast<int32_t>(n_tiles_dim1 * n_tiles_dim0) - 1)
            ? static_cast<int32_t>(n_isects)
            : tile_offsets[tile_id + 1];

    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory layout (same as forward)
    extern __shared__ int s[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(s);
    vec3<S>* xy_opacity_batch = reinterpret_cast<vec3<S>*>(&id_batch[block_size]);
    vec3<S>* conic_batch = reinterpret_cast<vec3<S>*>(&xy_opacity_batch[block_size]);
    S* mu_batch = reinterpret_cast<S*>(&conic_batch[block_size]);

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Process batches of Gaussians (front to back, order doesn't matter for CT)
    for (uint32_t b = 0; b < num_batches; ++b) {
        // Sync before loading next batch
        block.sync();

        // Each thread fetches 1 Gaussian
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + tr;
        if (idx < static_cast<uint32_t>(range_end)) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            mu_batch[tr] = mus[g];
        }

        // Wait for all threads to load
        block.sync();

        // Process Gaussians in the current batch
        const uint32_t batch_size = min(block_size, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; t < batch_size; ++t) {
            // Initialize local gradients
            vec2<S> v_xy_local = {static_cast<S>(0), static_cast<S>(0)};
            vec3<S> v_conic_local = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
            S v_opacity_local = static_cast<S>(0);
            S v_mu_local = static_cast<S>(0);

            bool valid = inside;
            if (valid) {
                const vec3<S> conic = conic_batch[t];
                const vec3<S> xy_opac = xy_opacity_batch[t];
                const S opac = xy_opac.z;
                const S mu = mu_batch[t];
                const vec2<S> delta = {px - xy_opac.x, py - xy_opac.y};

                // Compute sigma = 0.5 * delta^T * conic * delta
                const S sigma = static_cast<S>(0.5) * (
                    conic.x * delta.x * delta.x +
                    conic.z * delta.y * delta.y
                ) + conic.y * delta.x * delta.y;

                // Compute vis = exp(-sigma) and alpha = opacity * mu * vis
                const S vis = __expf(-sigma);
                const S alpha = opac * mu * vis;

                // Skip if sigma is negative or alpha is too small (same as forward)
                if (sigma < static_cast<S>(0) || fabsf(alpha) < static_cast<S>(1e-5)) {
                    valid = false;
                }

                if (valid) {
                    // v_sigma = -v_render * alpha
                    const S v_sigma = -v_render * alpha;

                    // Gradients w.r.t. opacity and mu
                    v_opacity_local = v_render * mu * vis;
                    v_mu_local = v_render * opac * vis;

                    // Gradients w.r.t. means2d
                    // δ = pixel - mean, so ∂δ/∂mean = -1
                    // sigma = 0.5 * (a*δx² + c*δy²) + b*δx*δy
                    // ∂sigma/∂δx = a*δx + b*δy
                    // ∂sigma/∂δy = c*δy + b*δx
                    // ∂sigma/∂mean_x = ∂sigma/∂δx * ∂δx/∂mean_x = (a*δx + b*δy) * (-1)
                    // ∂sigma/∂mean_y = ∂sigma/∂δy * ∂δy/∂mean_y = (c*δy + b*δx) * (-1)
                    // v_mean_x = v_sigma * ∂sigma/∂mean_x = -v_sigma * (a*δx + b*δy)
                    // v_mean_y = v_sigma * ∂sigma/∂mean_y = -v_sigma * (c*δy + b*δx)
                    v_xy_local.x = -v_sigma * (conic.x * delta.x + conic.y * delta.y);
                    v_xy_local.y = -v_sigma * (conic.z * delta.y + conic.y * delta.x);

                    // Gradients w.r.t. conics
                    // ∂sigma/∂a = 0.5 * δx²
                    // ∂sigma/∂b = δx * δy
                    // ∂sigma/∂c = 0.5 * δy²
                    v_conic_local.x = v_sigma * static_cast<S>(0.5) * delta.x * delta.x;
                    v_conic_local.y = v_sigma * delta.x * delta.y;
                    v_conic_local.z = v_sigma * static_cast<S>(0.5) * delta.y * delta.y;
                }
            }

            // Warp-level reduction
            gsplat::warpSum(v_xy_local, warp);
            gsplat::warpSum(v_conic_local, warp);
            gsplat::warpSum(v_opacity_local, warp);
            gsplat::warpSum(v_mu_local, warp);

            // First thread in warp atomically adds to global memory
            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[t];
                
                S* v_xy_ptr = reinterpret_cast<S*>(v_means2d) + 2 * g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);

                S* v_conic_ptr = reinterpret_cast<S*>(v_conics) + 3 * g;
                atomicAdd(v_conic_ptr, v_conic_local.x);
                atomicAdd(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 2, v_conic_local.z);

                atomicAdd(v_opacities + g, v_opacity_local);
                atomicAdd(v_mus + g, v_mu_local);
            }
        }
    }
}

// ============================================================================
// Launch Function
// ============================================================================

template <typename S>
void launch_rasterize_to_pixels_bwd(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means2d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_mus,
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
) {
    // Each block covers a tile on the image
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, n_tiles_dim0, n_tiles_dim1};

    // Calculate shared memory size (same as forward)
    const uint32_t block_size = tile_size * tile_size;
    const uint32_t shared_mem = block_size * (
        sizeof(int32_t) +
        sizeof(vec3<S>) +
        sizeof(vec3<S>) +
        sizeof(S)
    );

    // Set maximum shared memory size
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_bwd_kernel<S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_bwd_kernel<S><<<blocks, threads, shared_mem, stream>>>(
        C,
        N,
        n_isects,
        reinterpret_cast<const vec2<S>*>(means2d),
        reinterpret_cast<const vec3<S>*>(conics),
        opacities,
        mus,
        dim1,
        dim0,
        tile_size,
        n_tiles_dim1,
        n_tiles_dim0,
        tile_offsets,
        flatten_ids,
        v_render_output,
        reinterpret_cast<vec2<S>*>(v_means2d),
        reinterpret_cast<vec3<S>*>(v_conics),
        v_opacities,
        v_mus
    );
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template void launch_rasterize_to_pixels_bwd<float>(
    const float*, const float*, const float*, const float*,
    const int32_t*, const int32_t*,
    const float*,
    float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_rasterize_to_pixels_bwd<double>(
    const double*, const double*, const double*, const double*,
    const int32_t*, const int32_t*,
    const double*,
    double*, double*, double*, double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

// ============================================================================
// Second-Order Backward Pass Kernel (Hessian)
// ============================================================================

/**
 * Second-order backward pass for CT rasterization.
 * 
 * Computes both first-order gradients and second-order derivatives (Hessian).
 * 
 * For CT reconstruction: O = Σ_i G_i where G_i = μ_i × o_i × exp(-σ_i)
 * 
 * Given upstream:
 *   v_O = ∂L/∂O (first derivative)
 *   h_O = ∂²L/∂O² (second derivative / Hessian diagonal)
 * 
 * We compute using chain rule:
 *   v_θ = v_O × ∂G/∂θ
 *   h_{θi,θj} = h_O × (∂G/∂θi)(∂G/∂θj) + v_O × ∂²G/∂θi∂θj
 * 
 * Output Hessians store only upper triangular (independent) elements:
 *   h_means2d: [C*N, 3] upper triangle [H00, H01, H11]
 *   h_conics: [C*N, 6] upper triangle [H00, H01, H02, H11, H12, H22]
 *   h_mu_mean: [C*N, 2] (cross term μ-mean)
 *   h_mu_conic: [C*N, 3] (cross term μ-conic)
 *   h_mean_conic: [C*N, 2, 3] = [C*N, 6] (cross term mean-conic)
 * 
 * Storage order is consistent with conics [a, b, c] which stores upper triangle of
 * 2x2 inverse covariance matrix as row-major upper triangle.
 */
template <typename S>
__global__ void rasterize_to_pixels_bwd2_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    // Forward inputs
    const vec2<S>* __restrict__ means2d,    // [C * N, 2]
    const vec3<S>* __restrict__ conics,     // [C * N, 3]
    const S* __restrict__ opacities,        // [C * N]
    const S* __restrict__ mus,              // [C * N]
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0,
    const int32_t* __restrict__ tile_offsets,  // [C, n_tiles_dim0, n_tiles_dim1]
    const int32_t* __restrict__ flatten_ids,   // [n_isects]
    // Upstream gradients
    const S* __restrict__ v_render_output,     // [C, dim0, dim1]
    const S* __restrict__ h_render_output,     // [C, dim0, dim1]
    // Output first-order gradients
    vec2<S>* __restrict__ v_means2d,           // [C * N, 2]
    vec3<S>* __restrict__ v_conics,            // [C * N, 3]
    S* __restrict__ v_opacities,               // [C * N]
    S* __restrict__ v_mus,                     // [C * N]
    // Output second-order derivatives (upper triangle only)
    S* __restrict__ h_means2d,                 // [C * N, 3] upper triangle [H00, H01, H11]
    S* __restrict__ h_conics,                  // [C * N, 6] upper triangle [H00, H01, H02, H11, H12, H22]
    // Output cross-term Hessians
    vec2<S>* __restrict__ h_mu_mean,           // [C * N, 2]
    vec3<S>* __restrict__ h_mu_conic,          // [C * N, 3]
    S* __restrict__ h_mean_conic,              // [C * N, 2, 3] = [C * N, 6]
    S* __restrict__ h_mu_opacity,              // [C * N] (cross-term μ-opacity)
    vec3<S>* __restrict__ h_conic_opacity,     // [C * N, 3] (cross-term conic-opacity)
    vec2<S>* __restrict__ h_mean_opacity,      // [C * N, 2] (cross-term mean-opacity)
    // Output scalar Hessians
    S* __restrict__ h_mus,                     // [C * N]
    S* __restrict__ h_opacities                // [C * N]
) {
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * n_tiles_dim1 + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // Offset pointers for this camera
    tile_offsets += camera_id * n_tiles_dim0 * n_tiles_dim1;
    v_render_output += camera_id * dim0 * dim1;
    h_render_output += camera_id * dim0 * dim1;

    // Pixel center coordinates
    const S px = static_cast<S>(j) + static_cast<S>(0.5);
    const S py = static_cast<S>(i) + static_cast<S>(0.5);
    const int32_t pix_id = i * dim1 + j;

    // Check if inside valid pixel range
    const bool inside = (i < dim0 && j < dim1);

    // Get upstream gradients for this pixel
    const S v_O = inside ? v_render_output[pix_id] : static_cast<S>(0);
    const S h_O = inside ? h_render_output[pix_id] : static_cast<S>(0);

    // Which Gaussians to look through in this tile
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end = 
        (camera_id == static_cast<int32_t>(C) - 1) && 
        (tile_id == static_cast<int32_t>(n_tiles_dim1 * n_tiles_dim0) - 1)
            ? static_cast<int32_t>(n_isects)
            : tile_offsets[tile_id + 1];

    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory layout (same as bwd)
    extern __shared__ int s[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(s);
    vec3<S>* xy_opacity_batch = reinterpret_cast<vec3<S>*>(&id_batch[block_size]);
    vec3<S>* conic_batch = reinterpret_cast<vec3<S>*>(&xy_opacity_batch[block_size]);
    S* mu_batch = reinterpret_cast<S*>(&conic_batch[block_size]);

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Process batches of Gaussians
    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        // Each thread fetches 1 Gaussian
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + tr;
        if (idx < static_cast<uint32_t>(range_end)) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            mu_batch[tr] = mus[g];
        }

        block.sync();

        // Process Gaussians in the current batch
        const uint32_t batch_size = min(block_size, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; t < batch_size; ++t) {
            // Initialize local gradients and Hessians
            vec2<S> v_xy_local = {static_cast<S>(0), static_cast<S>(0)};
            vec3<S> v_conic_local = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
            S v_opacity_local = static_cast<S>(0);
            S v_mu_local = static_cast<S>(0);
            
            // Hessian matrices - store only upper triangular (independent) elements
            // h_means2d: 2x2 symmetric matrix, upper triangle stored as [H00, H01, H11]
            S h_xy_local[3] = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
            // h_conics: 3x3 symmetric matrix, upper triangle stored as [H00, H01, H02, H11, H12, H22]
            S h_conic_local[6] = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0),
                                   static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
            
            // Cross-term Hessians
            vec2<S> h_mu_mean_local = {static_cast<S>(0), static_cast<S>(0)};
            vec3<S> h_mu_conic_local = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
            S h_mean_conic_local[6] = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0),
                                        static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
            
            // Scalar Hessians (for mu and opacity)
            // h_μ = Σ h_O × (∂G/∂μ)² = Σ h_O × (oI)²
            // h_o = Σ h_O × (∂G/∂o)² = Σ h_O × (μI)²
            S h_mu_local = static_cast<S>(0);
            S h_opacity_local = static_cast<S>(0);
            
            // Cross-term Hessians involving opacity
            S h_mu_opacity_local = static_cast<S>(0);
            vec3<S> h_conic_opacity_local = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
            vec2<S> h_mean_opacity_local = {static_cast<S>(0), static_cast<S>(0)};

            bool valid = inside;
            if (valid) {
                const vec3<S> conic = conic_batch[t];
                const vec3<S> xy_opac = xy_opacity_batch[t];
                const S opac = xy_opac.z;
                const S mu = mu_batch[t];
                const vec2<S> delta = {px - xy_opac.x, py - xy_opac.y};

                // Compute sigma = 0.5 * delta^T * conic * delta
                const S sigma = static_cast<S>(0.5) * (
                    conic.x * delta.x * delta.x +
                    conic.z * delta.y * delta.y
                ) + conic.y * delta.x * delta.y;

                // Compute vis = exp(-sigma) and alpha = opacity * mu * vis
                const S vis = __expf(-sigma);
                const S alpha = opac * mu * vis;

                // Skip if sigma is negative or alpha is too small
                if (sigma < static_cast<S>(0) || fabsf(alpha) < static_cast<S>(1e-5)) {
                    valid = false;
                }

                if (valid) {
                    // =========================================================
                    // First-order derivatives (same as bwd)
                    // =========================================================
                    
                    // s = [a*δ1 + b*δ2, c*δ2 + b*δ1]
                    const vec2<S> s = {
                        conic.x * delta.x + conic.y * delta.y,
                        conic.z * delta.y + conic.y * delta.x
                    };
                    
                    // t = [0.5*δ1², δ1*δ2, 0.5*δ2²]
                    const vec3<S> t = {
                        static_cast<S>(0.5) * delta.x * delta.x,
                        delta.x * delta.y,
                        static_cast<S>(0.5) * delta.y * delta.y
                    };
                    
                    // v_sigma = -v_O * alpha
                    const S v_sigma = -v_O * alpha;
                    
                    // First-order gradients
                    v_opacity_local = v_O * mu * vis;
                    v_mu_local = v_O * opac * vis;
                    v_xy_local.x = -v_sigma * s.x;
                    v_xy_local.y = -v_sigma * s.y;
                    v_conic_local.x = v_sigma * t.x;
                    v_conic_local.y = v_sigma * t.y;
                    v_conic_local.z = v_sigma * t.z;
                    
                    // =========================================================
                    // Second-order derivatives
                    // =========================================================
                    // Chain rule for diagonal Hessian:
                    // h_{θi,θj} = Σ_pixels [ h_O[pixel] × (∂G/∂θi)(∂G/∂θj) + v_O[pixel] × ∂²G/∂θi∂θj ]
                    // 
                    // First-order derivatives (from bwd code):
                    //   ∂G/∂m = α×s  (positive, since v_xy = v_O×α×s)
                    //   ∂G/∂c = -α×t (negative, since v_conic = -v_O×α×t)
                    //   ∂G/∂μ = oI   (positive)
                    //   ∂G/∂o = μ×I  (positive)
                    
                    const S oI = opac * vis;              // o * I = opacity × exp(-σ)
                    
                    // h_means2d: ∂G/∂m = α×s
                    // h_{mi,mj} = h_O × (α×si)(α×sj) + v_O × ∂²G/∂mi∂mj
                    // ∂²G/∂m² = ∂/∂m(α×s) = s×∂α/∂m + α×∂s/∂m
                    //         = s×(α×s) + α×(-Σ⁻¹) = α×s×s - α×Σ⁻¹
                    // Thus: h_{mi,mj} = (h_O×α + v_O)×α×si×sj - v_O×α×Σ⁻¹[i,j]
                    const S ggn_m = (h_O * alpha + v_O) * alpha;
                    h_xy_local[0] = ggn_m * s.x * s.x - v_O * alpha * conic.x;  // H00
                    h_xy_local[1] = ggn_m * s.x * s.y - v_O * alpha * conic.y;  // H01
                    h_xy_local[2] = ggn_m * s.y * s.y - v_O * alpha * conic.z;  // H11
                    
                    // h_conics: ∂G/∂c = -α×t
                    // h_{ci,cj} = h_O × (-α×ti)(-α×tj) + v_O × ∂²G/∂ci∂cj
                    // ∂²G/∂c² = ∂/∂c(-α×t) = -t×∂α/∂c = -t×(-α×t) = α×t×t
                    // Thus: h_{ci,cj} = (h_O×α + v_O)×α×ti×tj
                    const S ggn_c = (h_O * alpha + v_O) * alpha;
                    h_conic_local[0] = ggn_c * t.x * t.x;  // H00
                    h_conic_local[1] = ggn_c * t.x * t.y;  // H01
                    h_conic_local[2] = ggn_c * t.x * t.z;  // H02
                    h_conic_local[3] = ggn_c * t.y * t.y;  // H11
                    h_conic_local[4] = ggn_c * t.y * t.z;  // H12
                    h_conic_local[5] = ggn_c * t.z * t.z;  // H22
                    
                    // Cross term h_mu_mean: ∂G/∂μ = oI, ∂G/∂m = α×s
                    // h_{μ,mi} = h_O × oI × (α×si) + v_O × ∂²G/∂μ∂mi
                    // ∂²G/∂μ∂m = ∂/∂μ(α×s) = oI×s
                    // Thus: h_{μ,mi} = (h_O×α + v_O) × oI × si
                    const S factor_mu = (h_O * alpha + v_O) * oI;
                    h_mu_mean_local.x = factor_mu * s.x;
                    h_mu_mean_local.y = factor_mu * s.y;
                    
                    // Cross term h_mu_conic: ∂G/∂μ = oI, ∂G/∂c = -α×t
                    // h_{μ,ci} = h_O × oI × (-α×ti) + v_O × ∂²G/∂μ∂ci
                    // ∂²G/∂μ∂c = ∂/∂μ(-α×t) = -oI×t
                    // Thus: h_{μ,ci} = -(h_O×α + v_O) × oI × ti
                    h_mu_conic_local.x = -factor_mu * t.x;
                    h_mu_conic_local.y = -factor_mu * t.y;
                    h_mu_conic_local.z = -factor_mu * t.z;
                    
                    // Cross term h_mean_conic: ∂G/∂m = α×s, ∂G/∂c = -α×t
                    // h_{mi,cj} = h_O × (α×si) × (-α×tj) + v_O × ∂²G/∂mi∂cj
                    //           = -h_O×α²×si×tj + v_O × ∂²G/∂mi∂cj
                    // ∂²G/∂m∂c = ∂/∂c(α×s) = s×∂α/∂c + α×∂s/∂c
                    //          = s×(-α×t) + α×∂s/∂c = -α×s×t + α×∂s/∂c
                    // where s = [a×δx+b×δy, c×δy+b×δx], ∂s/∂c gives:
                    // ∂s1/∂a = δx, ∂s1/∂b = δy, ∂s1/∂c = 0
                    // ∂s2/∂a = 0,  ∂s2/∂b = δx, ∂s2/∂c = δy
                    // Thus: h_{mi,cj} = -h_O×α²×si×tj + v_O×(-α×si×tj + α×∂si/∂cj)
                    //                 = -(h_O×α + v_O)×α×si×tj + v_O×α×∂si/∂cj
                    const S common_mc = -(h_O * alpha + v_O) * alpha;
                    h_mean_conic_local[0] = common_mc * s.x * t.x + v_O * alpha * delta.x;  // ∂²/∂m1∂a
                    h_mean_conic_local[1] = common_mc * s.x * t.y + v_O * alpha * delta.y;  // ∂²/∂m1∂b
                    h_mean_conic_local[2] = common_mc * s.x * t.z;                          // ∂²/∂m1∂c
                    h_mean_conic_local[3] = common_mc * s.y * t.x;                          // ∂²/∂m2∂a
                    h_mean_conic_local[4] = common_mc * s.y * t.y + v_O * alpha * delta.x;  // ∂²/∂m2∂b
                    h_mean_conic_local[5] = common_mc * s.y * t.z + v_O * alpha * delta.y;  // ∂²/∂m2∂c
                    
                    // =========================================================
                    // Scalar Hessians for mu and opacity
                    // =========================================================
                    // h_μ = h_O × (∂G/∂μ)² + v_O × ∂²G/∂μ²
                    //     where ∂G/∂μ = oI = opacity × exp(-σ), ∂²G/∂μ² = 0
                    //     thus h_μ = h_O × oI²
                    //
                    // h_o = h_O × (∂G/∂o)² + v_O × ∂²G/∂o²
                    //     where ∂G/∂o = μI = mu × exp(-σ), ∂²G/∂o² = 0
                    //     thus h_o = h_O × (μI)²
                    const S muI = mu * vis;  // mu × exp(-σ)
                    h_mu_local = h_O * oI * oI;
                    h_opacity_local = h_O * muI * muI;
                    
                    // =========================================================
                    // Cross-term Hessians involving opacity
                    // =========================================================
                    // factor_opac = (h_O * α + v_O) * muI
                    const S factor_opac = (h_O * alpha + v_O) * muI;
                    
                    // h_{μ,o} = h_O × (oI)(muI) + v_O × vis
                    h_mu_opacity_local = h_O * oI * muI + v_O * vis;
                    
                    // h_{c_j,o} = -(h_O × α + v_O) × muI × t_j
                    h_conic_opacity_local.x = -factor_opac * t.x;
                    h_conic_opacity_local.y = -factor_opac * t.y;
                    h_conic_opacity_local.z = -factor_opac * t.z;
                    
                    // h_{m_i,o} = (h_O × α + v_O) × muI × s_i
                    h_mean_opacity_local.x = factor_opac * s.x;
                    h_mean_opacity_local.y = factor_opac * s.y;
                }
            }

            // Warp-level reduction for first-order gradients
            gsplat::warpSum(v_xy_local, warp);
            gsplat::warpSum(v_conic_local, warp);
            gsplat::warpSum(v_opacity_local, warp);
            gsplat::warpSum(v_mu_local, warp);
            
            // Warp-level reduction for second-order derivatives (upper triangle only)
            // h_means2d: 3 elements [H00, H01, H11]
            gsplat::warpSum<3>(h_xy_local, warp);
            // h_conics: 6 elements [H00, H01, H02, H11, H12, H22]
            gsplat::warpSum<6>(h_conic_local, warp);
            
            gsplat::warpSum(h_mu_mean_local, warp);
            gsplat::warpSum(h_mu_conic_local, warp);
            
            // Reduce h_mean_conic_local: 6 elements
            gsplat::warpSum<6>(h_mean_conic_local, warp);
            
            // Reduce scalar Hessians
            gsplat::warpSum(h_mu_local, warp);
            gsplat::warpSum(h_opacity_local, warp);
            
            // Reduce new cross-term Hessians involving opacity
            gsplat::warpSum(h_mu_opacity_local, warp);
            gsplat::warpSum(h_conic_opacity_local, warp);
            gsplat::warpSum(h_mean_opacity_local, warp);

            // First thread in warp atomically adds to global memory
            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[t];
                
                // First-order gradients
                S* v_xy_ptr = reinterpret_cast<S*>(v_means2d) + 2 * g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);

                S* v_conic_ptr = reinterpret_cast<S*>(v_conics) + 3 * g;
                atomicAdd(v_conic_ptr, v_conic_local.x);
                atomicAdd(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 2, v_conic_local.z);

                atomicAdd(v_opacities + g, v_opacity_local);
                atomicAdd(v_mus + g, v_mu_local);
                
                // Second-order derivatives (upper triangle only)
                // h_means2d: [C*N, 3] upper triangle [H00, H01, H11]
                S* h_xy_ptr = h_means2d + 3 * g;
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    atomicAdd(h_xy_ptr + k, h_xy_local[k]);
                }
                
                // h_conics: [C*N, 6] upper triangle [H00, H01, H02, H11, H12, H22]
                S* h_conic_ptr = h_conics + 6 * g;
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    atomicAdd(h_conic_ptr + k, h_conic_local[k]);
                }
                
                // Cross-term Hessians
                S* h_mu_mean_ptr = reinterpret_cast<S*>(h_mu_mean) + 2 * g;
                atomicAdd(h_mu_mean_ptr, h_mu_mean_local.x);
                atomicAdd(h_mu_mean_ptr + 1, h_mu_mean_local.y);
                
                S* h_mu_conic_ptr = reinterpret_cast<S*>(h_mu_conic) + 3 * g;
                atomicAdd(h_mu_conic_ptr, h_mu_conic_local.x);
                atomicAdd(h_mu_conic_ptr + 1, h_mu_conic_local.y);
                atomicAdd(h_mu_conic_ptr + 2, h_mu_conic_local.z);
                
                // h_mean_conic: [C*N, 6] (2x3 flattened)
                S* h_mc_ptr = h_mean_conic + 6 * g;
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    atomicAdd(h_mc_ptr + k, h_mean_conic_local[k]);
                }
                
                // Scalar Hessians
                atomicAdd(h_mus + g, h_mu_local);
                atomicAdd(h_opacities + g, h_opacity_local);
                
                // New cross-term Hessians involving opacity
                atomicAdd(h_mu_opacity + g, h_mu_opacity_local);
                
                S* h_co_ptr = reinterpret_cast<S*>(h_conic_opacity) + 3 * g;
                atomicAdd(h_co_ptr, h_conic_opacity_local.x);
                atomicAdd(h_co_ptr + 1, h_conic_opacity_local.y);
                atomicAdd(h_co_ptr + 2, h_conic_opacity_local.z);
                
                S* h_mo_ptr = reinterpret_cast<S*>(h_mean_opacity) + 2 * g;
                atomicAdd(h_mo_ptr, h_mean_opacity_local.x);
                atomicAdd(h_mo_ptr + 1, h_mean_opacity_local.y);
            }
        }
    }
}

// ============================================================================
// Second-Order Backward Launch Function
// ============================================================================

template <typename S>
void launch_rasterize_to_pixels_bwd2(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    const S* __restrict__ h_render_output,
    S* __restrict__ v_means2d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_mus,
    S* __restrict__ h_means2d,
    S* __restrict__ h_conics,
    S* __restrict__ h_mu_mean,
    S* __restrict__ h_mu_conic,
    S* __restrict__ h_mean_conic,
    S* __restrict__ h_mu_opacity,
    S* __restrict__ h_conic_opacity,
    S* __restrict__ h_mean_opacity,
    S* __restrict__ h_mus,
    S* __restrict__ h_opacities,
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
) {
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, n_tiles_dim0, n_tiles_dim1};

    const uint32_t block_size = tile_size * tile_size;
    const uint32_t shared_mem = block_size * (
        sizeof(int32_t) +
        sizeof(vec3<S>) +
        sizeof(vec3<S>) +
        sizeof(S)
    );

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_bwd2_kernel<S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_bwd2_kernel<S><<<blocks, threads, shared_mem, stream>>>(
        C,
        N,
        n_isects,
        reinterpret_cast<const vec2<S>*>(means2d),
        reinterpret_cast<const vec3<S>*>(conics),
        opacities,
        mus,
        dim1,
        dim0,
        tile_size,
        n_tiles_dim1,
        n_tiles_dim0,
        tile_offsets,
        flatten_ids,
        v_render_output,
        h_render_output,
        reinterpret_cast<vec2<S>*>(v_means2d),
        reinterpret_cast<vec3<S>*>(v_conics),
        v_opacities,
        v_mus,
        h_means2d,
        h_conics,
        reinterpret_cast<vec2<S>*>(h_mu_mean),
        reinterpret_cast<vec3<S>*>(h_mu_conic),
        h_mean_conic,
        h_mu_opacity,
        reinterpret_cast<vec3<S>*>(h_conic_opacity),
        reinterpret_cast<vec2<S>*>(h_mean_opacity),
        h_mus,
        h_opacities
    );
}

// Second-order template instantiations
template void launch_rasterize_to_pixels_bwd2<float>(
    const float*, const float*, const float*, const float*,
    const int32_t*, const int32_t*,
    const float*, const float*,
    float*, float*, float*, float*,
    float*, float*, float*, float*, float*, float*, float*, float*,
    float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_rasterize_to_pixels_bwd2<double>(
    const double*, const double*, const double*, const double*,
    const int32_t*, const int32_t*,
    const double*, const double*,
    double*, double*, double*, double*,
    double*, double*, double*, double*, double*, double*, double*, double*,
    double*, double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

}  // namespace gsplat2ct
