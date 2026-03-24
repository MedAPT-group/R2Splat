/*NOTES by Dy164:
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
Keep the following correspondance in mind:
    - px -- k -- dim2 -- x (fastest varying, columns)
    - py -- j -- dim1 -- y (middle dimension)
    - pz -- i -- dim0 -- z (slowest varying, rows)
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
*/
/**
 * VoxelizeToVoxels Backward CUDA Kernels
 * 
 * This file implements CUDA kernels for tile-based voxelization backward pass
 * specifically designed for CT reconstruction.
 * 
 * Key features:
 * - Single channel output (grayscale for CT)
 * - No alpha blending: CT uses simple summation, order-independent
 * - Supports both first-order gradients and second-order Hessian
 * - Tile-based with shared memory optimization
 * 
 * CT forward formula: voxel_value = Σ (opacity × exp(-sigma))
 * where sigma = 0.5 × delta^T × conic × delta
 * 
 * Gradient notation:
 * - v_* : first-order gradient (∂L/∂*)
 * - h_* : second-order Hessian (∂²L/∂*²)
 * 
 * Conic storage: [N, 6] = [a, p, q, b, r, c] (upper triangular of 3x3 matrix)
 *   [[a, p, q],
 *    [p, b, r],
 *    [q, r, c]]
 * 
 * Author: Dy164
 * Date: 2026-02-03
 */

#include "VoxelizeToVoxels.h"
#include "../include/types.cuh"
#include "../include/helpers.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace gsplat2ct {

namespace cg = cooperative_groups;

// ============================================================================
// First-Order Backward Pass Kernel
// ============================================================================

/**
 * First-order backward pass for CT voxelization.
 * 
 * For CT reconstruction (simple summation, no alpha blending):
 *   O = Σ_g G_g where G = o × exp(-σ)
 *   
 * Given v_render = ∂L/∂O from upstream, we compute:
 *   v_opacity = v_O × exp(-σ)
 *   v_sigma = -v_O × o × exp(-σ) = -v_O × G
 *   
 * Then expand sigma = 0.5×(a×d1² + b×d2² + c×d3²) + p×d1×d2 + q×d1×d3 + r×d2×d3:
 *   v_mean = v_sigma × [a×d1+p×d2+q×d3, b×d2+p×d1+r×d3, c×d3+q×d1+r×d2]
 *   v_conic = v_sigma × [0.5×d1², d1×d2, d1×d3, 0.5×d2², d2×d3, 0.5×d3²]
 *           = v_sigma × [c00, c01, c02, c11, c12, c22] (storage order)
 */
template <typename S>
__global__ void voxelize_to_voxels_bwd_kernel(
    const uint32_t N,
    const uint32_t n_isects,
    // Forward inputs
    const vec3<S>* __restrict__ means3d,    // [N, 3]
    const S* __restrict__ conics,           // [N, 6]
    const S* __restrict__ opacities,        // [N]
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0,
    const int32_t* __restrict__ tile_offsets,  // [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
    const int32_t* __restrict__ flatten_ids,   // [n_isects]
    // Upstream gradient
    const S* __restrict__ v_render_output,     // [dim0, dim1, dim2]
    // Output gradients
    vec3<S>* __restrict__ v_means3d,           // [N, 3]
    S* __restrict__ v_conics,                  // [N, 6]
    S* __restrict__ v_opacities                // [N]
) {
    auto block = cg::this_thread_block();
    
    int32_t tile_id = block.group_index().x * n_tiles_dim1 * n_tiles_dim2 +
                      block.group_index().y * n_tiles_dim2 +
                      block.group_index().z;
    
    uint32_t i = block.group_index().x * tile_size + block.thread_index().x;  // dim0
    uint32_t j = block.group_index().y * tile_size + block.thread_index().y;  // dim1
    uint32_t k = block.group_index().z * tile_size + block.thread_index().z;  // dim2

    // Voxel center coordinates
    const S px = static_cast<S>(k) + static_cast<S>(0.5);
    const S py = static_cast<S>(j) + static_cast<S>(0.5);
    const S pz = static_cast<S>(i) + static_cast<S>(0.5);
    const int32_t vox_id = i * dim1 * dim2 + j * dim2 + k;

    const bool inside = (i < dim0 && j < dim1 && k < dim2);

    // Get upstream gradient
    const S v_O = inside ? v_render_output[vox_id] : static_cast<S>(0);

    // Tile range
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end = (tile_id == static_cast<int32_t>(n_tiles_dim0 * n_tiles_dim1 * n_tiles_dim2) - 1)
                                  ? static_cast<int32_t>(n_isects)
                                  : tile_offsets[tile_id + 1];
    
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory layout
    extern __shared__ int s[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(s);
    vec3<S>* xyz_batch = reinterpret_cast<vec3<S>*>(&id_batch[block_size]);
    S* conic_batch = reinterpret_cast<S*>(&xyz_batch[block_size]);
    S* opac_batch = &conic_batch[block_size * 6];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        // Load Gaussian data
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < static_cast<uint32_t>(range_end)) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xyz_batch[tr] = means3d[g];
            opac_batch[tr] = opacities[g];
            const S* conic_ptr = conics + g * 6;
            S* dest_ptr = conic_batch + tr * 6;
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                dest_ptr[c] = conic_ptr[c];
            }
        }
        block.sync();

        uint32_t batch_size = min(block_size, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; t < batch_size; ++t) {
            bool valid = inside;
            S opac, vis;
            vec3<S> delta;
            S conic[6];

            if (valid) {
                vec3<S> xyz = xyz_batch[t];
                opac = opac_batch[t];
                const S* conic_ptr = conic_batch + t * 6;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    conic[c] = conic_ptr[c];
                }

                delta = {px - xyz.x, py - xyz.y, pz - xyz.z};
                
                // sigma = 0.5*(a*d1^2 + b*d2^2 + c*d3^2) + p*d1*d2 + q*d1*d3 + r*d2*d3
                // conic = [a, p, q, b, r, c]
                const S a = conic[0], p = conic[1], q = conic[2];
                const S bb = conic[3], r = conic[4], c = conic[5];
                const S d1 = delta.x, d2 = delta.y, d3 = delta.z;
                
                const S sigma = static_cast<S>(0.5) * (a*d1*d1 + bb*d2*d2 + c*d3*d3) +
                                p*d1*d2 + q*d1*d3 + r*d2*d3;
                vis = __expf(-sigma);
                const S alpha = opac * vis;
                
                if (sigma < static_cast<S>(0) || fabsf(alpha) < static_cast<S>(1e-5)) {
                    valid = false;
                }
            }

            if (!warp.any(valid)) {
                continue;
            }

            // Initialize local gradients
            vec3<S> v_xyz_local = {0, 0, 0};
            S v_conic_local[6] = {0, 0, 0, 0, 0, 0};
            S v_opacity_local = 0;

            if (valid) {
                const S a = conic[0], p = conic[1], q = conic[2];
                const S bb = conic[3], r = conic[4], c = conic[5];
                const S d1 = delta.x, d2 = delta.y, d3 = delta.z;

                // v_sigma = -v_O * o * vis (used for conic gradients)
                const S v_sigma = -v_O * opac * vis;
                
                // oI = o * vis (used for mean gradients)
                const S oI = opac * vis;

                // v_opacity = v_O * vis
                v_opacity_local = v_O * vis;

                // v_conic = v_sigma * [0.5*d1^2, d1*d2, d1*d3, 0.5*d2^2, d2*d3, 0.5*d3^2]
                v_conic_local[0] = v_sigma * static_cast<S>(0.5) * d1 * d1;  // a
                v_conic_local[1] = v_sigma * d1 * d2;                        // p
                v_conic_local[2] = v_sigma * d1 * d3;                        // q
                v_conic_local[3] = v_sigma * static_cast<S>(0.5) * d2 * d2;  // b
                v_conic_local[4] = v_sigma * d2 * d3;                        // r
                v_conic_local[5] = v_sigma * static_cast<S>(0.5) * d3 * d3;  // c

                // v_mean = v_O * oI * s (where s = Σ^{-1} @ d)
                // Note: dG/dm = oI * s (positive sign, not v_sigma!)
                v_xyz_local.x = v_O * oI * (a*d1 + p*d2 + q*d3);
                v_xyz_local.y = v_O * oI * (bb*d2 + p*d1 + r*d3);
                v_xyz_local.z = v_O * oI * (c*d3 + q*d1 + r*d2);
            }

            // Warp reduction
            gsplat::warpSum(v_xyz_local, warp);
            gsplat::warpSum<6>(v_conic_local, warp);
            gsplat::warpSum(v_opacity_local, warp);

            // Atomic add to global memory (only warp leader)
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                
                S* v_xyz_ptr = reinterpret_cast<S*>(v_means3d) + 3 * g;
                atomicAdd(v_xyz_ptr, v_xyz_local.x);
                atomicAdd(v_xyz_ptr + 1, v_xyz_local.y);
                atomicAdd(v_xyz_ptr + 2, v_xyz_local.z);

                S* v_conic_ptr = v_conics + 6 * g;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    atomicAdd(v_conic_ptr + c, v_conic_local[c]);
                }

                atomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

// ============================================================================
// Second-Order Backward Pass Kernel
// ============================================================================

/**
 * Second-order backward pass for CT voxelization.
 * 
 * Computes both first-order gradients and second-order Hessians.
 * 
 * For CT: G = o × exp(-σ) where σ = 0.5 × d^T × Σ^{-1} × d
 * 
 * Given upstream v_O = ∂L/∂O and h_O = ∂²L/∂O² (diagonal approximation),
 * we compute gradients and Hessians using chain rule:
 *   v_θ = v_O × ∂G/∂θ
 *   H_{θi,θj} = h_O × (∂G/∂θi)(∂G/∂θj) + v_O × ∂²G/∂θi∂θj
 * 
 * Hessian storage (upper triangular):
 *   h_means3d: [N, 6] = [H00, H01, H02, H11, H12, H22]
 *   h_conics: [N, 21] = upper triangle of 6×6 matrix
 *   h_opacities: [N] (scalar)
 *   h_op_mean: [N, 3] (cross term opacity-means3d)
 *   h_op_conic: [N, 6] (cross term opacity-conics)
 *   h_mean_conic: [N, 18] = [N, 3, 6] (cross term means3d-conics)
 */
template <typename S>
__global__ void voxelize_to_voxels_bwd2_kernel(
    const uint32_t N,
    const uint32_t n_isects,
    // Forward inputs
    const vec3<S>* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    // Upstream gradients
    const S* __restrict__ v_render_output,     // [dim0, dim1, dim2]
    const S* __restrict__ h_render_output,     // [dim0, dim1, dim2]
    // Output first-order gradients
    vec3<S>* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    // Output Hessians (diagonal blocks)
    S* __restrict__ h_means3d,      // [N, 6] upper triangle
    S* __restrict__ h_conics,       // [N, 21] upper triangle
    S* __restrict__ h_opacities,    // [N]
    // Output Hessians (cross terms)
    S* __restrict__ h_op_mean,      // [N, 3]
    S* __restrict__ h_op_conic,     // [N, 6]
    S* __restrict__ h_mean_conic    // [N, 18]
) {
    auto block = cg::this_thread_block();
    
    int32_t tile_id = block.group_index().x * n_tiles_dim1 * n_tiles_dim2 +
                      block.group_index().y * n_tiles_dim2 +
                      block.group_index().z;
    
    uint32_t i = block.group_index().x * tile_size + block.thread_index().x;
    uint32_t j = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t k = block.group_index().z * tile_size + block.thread_index().z;

    const S px = static_cast<S>(k) + static_cast<S>(0.5);
    const S py = static_cast<S>(j) + static_cast<S>(0.5);
    const S pz = static_cast<S>(i) + static_cast<S>(0.5);
    const int32_t vox_id = i * dim1 * dim2 + j * dim2 + k;

    const bool inside = (i < dim0 && j < dim1 && k < dim2);

    const S v_O = inside ? v_render_output[vox_id] : static_cast<S>(0);
    const S h_O = inside ? h_render_output[vox_id] : static_cast<S>(0);

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end = (tile_id == static_cast<int32_t>(n_tiles_dim0 * n_tiles_dim1 * n_tiles_dim2) - 1)
                                  ? static_cast<int32_t>(n_isects)
                                  : tile_offsets[tile_id + 1];
    
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(s);
    vec3<S>* xyz_batch = reinterpret_cast<vec3<S>*>(&id_batch[block_size]);
    S* conic_batch = reinterpret_cast<S*>(&xyz_batch[block_size]);
    S* opac_batch = &conic_batch[block_size * 6];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < static_cast<uint32_t>(range_end)) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xyz_batch[tr] = means3d[g];
            opac_batch[tr] = opacities[g];
            const S* conic_ptr = conics + g * 6;
            S* dest_ptr = conic_batch + tr * 6;
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                dest_ptr[c] = conic_ptr[c];
            }
        }
        block.sync();

        uint32_t batch_size = min(block_size, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; t < batch_size; ++t) {
            bool valid = inside;
            S opac, vis;
            S conic[6];
            S d1, d2, d3;

            if (valid) {
                vec3<S> xyz = xyz_batch[t];
                opac = opac_batch[t];
                const S* conic_ptr = conic_batch + t * 6;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    conic[c] = conic_ptr[c];
                }

                d1 = px - xyz.x;
                d2 = py - xyz.y;
                d3 = pz - xyz.z;
                
                const S a = conic[0], p = conic[1], q = conic[2];
                const S bb = conic[3], r = conic[4], c = conic[5];
                
                const S sigma = static_cast<S>(0.5) * (a*d1*d1 + bb*d2*d2 + c*d3*d3) +
                                p*d1*d2 + q*d1*d3 + r*d2*d3;
                vis = __expf(-sigma);
                const S alpha = opac * vis;
                
                if (sigma < static_cast<S>(0) || fabsf(alpha) < static_cast<S>(1e-5)) {
                    valid = false;
                }
            }

            if (!warp.any(valid)) {
                continue;
            }

            // Initialize local accumulators
            // First-order gradients
            vec3<S> v_xyz_local = {0, 0, 0};
            S v_conic_local[6] = {0, 0, 0, 0, 0, 0};
            S v_opacity_local = 0;
            
            // Second-order Hessians
            S h_means3d_local[6] = {0, 0, 0, 0, 0, 0};  // upper triangle 3x3
            S h_conics_local[21] = {0};                  // upper triangle 6x6
            S h_opacities_local = 0;
            S h_op_mean_local[3] = {0, 0, 0};
            S h_op_conic_local[6] = {0, 0, 0, 0, 0, 0};
            S h_mean_conic_local[18] = {0};              // 3x6 matrix

            if (valid) {
                const S a = conic[0], p = conic[1], q = conic[2];
                const S bb = conic[3], r = conic[4], c = conic[5];
                const S oI = opac * vis;  // o × I

                // s = ∂σ/∂d = Σ^{-1} × d
                const S s1 = a*d1 + p*d2 + q*d3;
                const S s2 = bb*d2 + p*d1 + r*d3;
                const S s3 = c*d3 + q*d1 + r*d2;

                // v = ∂σ/∂conic = [c00, c01, c02, c11, c12, c22]
                // Matches storage order: [a, p, q, b, r, c]
                const S v_arr[6] = {
                    static_cast<S>(0.5)*d1*d1,  // c00 = a
                    d1*d2,                      // c01 = p
                    d1*d3,                      // c02 = q
                    static_cast<S>(0.5)*d2*d2,  // c11 = b
                    d2*d3,                      // c12 = r
                    static_cast<S>(0.5)*d3*d3   // c22 = c
                };

                // ============================================================
                // First-order gradients
                // ============================================================
                // dG/do = I
                const S dG_do = vis;
                // dG/dm = oI × s
                const S dG_dm[3] = {oI * s1, oI * s2, oI * s3};
                // dG/dconic = -oI × v
                S dG_dconic[6];
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    dG_dconic[c] = -oI * v_arr[c];
                }

                v_opacity_local = v_O * dG_do;
                v_xyz_local = {v_O * dG_dm[0], v_O * dG_dm[1], v_O * dG_dm[2]};
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    v_conic_local[c] = v_O * dG_dconic[c];
                }

                // ============================================================
                // Second-order Hessians
                // H = h_O × (∂G/∂θi)(∂G/∂θj) + v_O × ∂²G/∂θi∂θj
                // ============================================================

                // --- h_opacities: d²G/do² = 0 ---
                // Only Gauss-Newton term
                h_opacities_local = h_O * dG_do * dG_do;

                // --- h_means3d: d²G/dm² = oI(ss^T - Σ^{-1}) ---
                // Σ^{-1} = [[a,p,q],[p,b,r],[q,r,c]]
                // Upper triangle storage: [H00, H01, H02, H11, H12, H22]
                const S d2G_dm2[6] = {
                    oI * (s1*s1 - a),    // H00
                    oI * (s1*s2 - p),    // H01
                    oI * (s1*s3 - q),    // H02
                    oI * (s2*s2 - bb),   // H11
                    oI * (s2*s3 - r),    // H12
                    oI * (s3*s3 - c)     // H22
                };
                // Full Hessian (upper triangle)
                h_means3d_local[0] = h_O * dG_dm[0] * dG_dm[0] + v_O * d2G_dm2[0];
                h_means3d_local[1] = h_O * dG_dm[0] * dG_dm[1] + v_O * d2G_dm2[1];
                h_means3d_local[2] = h_O * dG_dm[0] * dG_dm[2] + v_O * d2G_dm2[2];
                h_means3d_local[3] = h_O * dG_dm[1] * dG_dm[1] + v_O * d2G_dm2[3];
                h_means3d_local[4] = h_O * dG_dm[1] * dG_dm[2] + v_O * d2G_dm2[4];
                h_means3d_local[5] = h_O * dG_dm[2] * dG_dm[2] + v_O * d2G_dm2[5];

                // --- h_conics: d²G/dconic² = oI × vv^T ---
                // Upper triangle of 6x6: 21 elements
                int idx = 0;
                #pragma unroll
                for (int ii = 0; ii < 6; ++ii) {
                    #pragma unroll
                    for (int jj = ii; jj < 6; ++jj) {
                        const S d2G_dconic2 = oI * v_arr[ii] * v_arr[jj];
                        h_conics_local[idx] = h_O * dG_dconic[ii] * dG_dconic[jj] + v_O * d2G_dconic2;
                        ++idx;
                    }
                }

                // --- h_op_mean: d²G/dodm = I × s ---
                const S d2G_dodm[3] = {vis * s1, vis * s2, vis * s3};
                h_op_mean_local[0] = h_O * dG_do * dG_dm[0] + v_O * d2G_dodm[0];
                h_op_mean_local[1] = h_O * dG_do * dG_dm[1] + v_O * d2G_dodm[1];
                h_op_mean_local[2] = h_O * dG_do * dG_dm[2] + v_O * d2G_dodm[2];

                // --- h_op_conic: d²G/dodconic = -I × v ---
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    const S d2G_doconic = -vis * v_arr[c];
                    h_op_conic_local[c] = h_O * dG_do * dG_dconic[c] + v_O * d2G_doconic;
                }

                // --- h_mean_conic: d²G/dmdconic = oI(J - s×v^T) ---
                // J = ∂s/∂conic where s = ∂σ/∂d and conic=[a,p,q,b,r,c].
                // Storage: row-major [3, 6]
                const S J[3][6] = {
                    {d1, d2, d3, 0,  0,  0},
                    {0,  d1, 0,  d2, d3, 0},
                    {0,  0,  d1, 0,  d2, d3}
                };
                const S s_arr[3] = {s1, s2, s3};
                
                idx = 0;
                #pragma unroll
                for (int mi = 0; mi < 3; ++mi) {
                    #pragma unroll
                    for (int ci = 0; ci < 6; ++ci) {
                        const S d2G_dmdc = oI * (J[mi][ci] - s_arr[mi] * v_arr[ci]);
                        h_mean_conic_local[idx] = h_O * dG_dm[mi] * dG_dconic[ci] + v_O * d2G_dmdc;
                        ++idx;
                    }
                }
            }

            // Warp reduction for all accumulators
            gsplat::warpSum(v_xyz_local, warp);
            gsplat::warpSum<6>(v_conic_local, warp);
            gsplat::warpSum(v_opacity_local, warp);
            
            gsplat::warpSum<6>(h_means3d_local, warp);
            gsplat::warpSum<21>(h_conics_local, warp);
            gsplat::warpSum(h_opacities_local, warp);
            gsplat::warpSum<3>(h_op_mean_local, warp);
            gsplat::warpSum<6>(h_op_conic_local, warp);
            gsplat::warpSum<18>(h_mean_conic_local, warp);

            // Atomic add to global memory
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                
                // First-order gradients
                S* v_xyz_ptr = reinterpret_cast<S*>(v_means3d) + 3 * g;
                atomicAdd(v_xyz_ptr, v_xyz_local.x);
                atomicAdd(v_xyz_ptr + 1, v_xyz_local.y);
                atomicAdd(v_xyz_ptr + 2, v_xyz_local.z);

                S* v_conic_ptr = v_conics + 6 * g;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    atomicAdd(v_conic_ptr + c, v_conic_local[c]);
                }
                atomicAdd(v_opacities + g, v_opacity_local);

                // Second-order Hessians
                S* h_m_ptr = h_means3d + 6 * g;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    atomicAdd(h_m_ptr + c, h_means3d_local[c]);
                }

                S* h_c_ptr = h_conics + 21 * g;
                #pragma unroll
                for (int c = 0; c < 21; ++c) {
                    atomicAdd(h_c_ptr + c, h_conics_local[c]);
                }

                atomicAdd(h_opacities + g, h_opacities_local);

                S* h_om_ptr = h_op_mean + 3 * g;
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    atomicAdd(h_om_ptr + c, h_op_mean_local[c]);
                }

                S* h_oc_ptr = h_op_conic + 6 * g;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    atomicAdd(h_oc_ptr + c, h_op_conic_local[c]);
                }

                S* h_mc_ptr = h_mean_conic + 18 * g;
                #pragma unroll
                for (int c = 0; c < 18; ++c) {
                    atomicAdd(h_mc_ptr + c, h_mean_conic_local[c]);
                }
            }
        }
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

template <typename S>
void launch_voxelize_to_voxels_bwd(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
) {
    dim3 threads = {tile_size, tile_size, tile_size};
    dim3 blocks = {n_tiles_dim0, n_tiles_dim1, n_tiles_dim2};

    const uint32_t block_size = tile_size * tile_size * tile_size;
    const uint32_t shared_mem = block_size * (
        sizeof(int32_t) +
        sizeof(vec3<S>) +
        6 * sizeof(S) +
        sizeof(S)
    );

    if (cudaFuncSetAttribute(
            voxelize_to_voxels_bwd_kernel<S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        // Fallback
    }

    voxelize_to_voxels_bwd_kernel<S><<<blocks, threads, shared_mem, stream>>>(
        N,
        n_isects,
        reinterpret_cast<const vec3<S>*>(means3d),
        conics,
        opacities,
        dim2,
        dim1,
        dim0,
        tile_size,
        n_tiles_dim2,
        n_tiles_dim1,
        n_tiles_dim0,
        tile_offsets,
        flatten_ids,
        v_render_output,
        reinterpret_cast<vec3<S>*>(v_means3d),
        v_conics,
        v_opacities
    );
}

template <typename S>
void launch_voxelize_to_voxels_bwd2(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    const S* __restrict__ h_render_output,
    S* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ h_means3d,
    S* __restrict__ h_conics,
    S* __restrict__ h_opacities,
    S* __restrict__ h_op_mean,
    S* __restrict__ h_op_conic,
    S* __restrict__ h_mean_conic,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
) {
    dim3 threads = {tile_size, tile_size, tile_size};
    dim3 blocks = {n_tiles_dim0, n_tiles_dim1, n_tiles_dim2};

    const uint32_t block_size = tile_size * tile_size * tile_size;
    const uint32_t shared_mem = block_size * (
        sizeof(int32_t) +
        sizeof(vec3<S>) +
        6 * sizeof(S) +
        sizeof(S)
    );

    if (cudaFuncSetAttribute(
            voxelize_to_voxels_bwd2_kernel<S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        // Fallback
    }

    voxelize_to_voxels_bwd2_kernel<S><<<blocks, threads, shared_mem, stream>>>(
        N,
        n_isects,
        reinterpret_cast<const vec3<S>*>(means3d),
        conics,
        opacities,
        dim2,
        dim1,
        dim0,
        tile_size,
        n_tiles_dim2,
        n_tiles_dim1,
        n_tiles_dim0,
        tile_offsets,
        flatten_ids,
        v_render_output,
        h_render_output,
        reinterpret_cast<vec3<S>*>(v_means3d),
        v_conics,
        v_opacities,
        h_means3d,
        h_conics,
        h_opacities,
        h_op_mean,
        h_op_conic,
        h_mean_conic
    );
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template void launch_voxelize_to_voxels_bwd<float>(
    const float*, const float*, const float*,
    const int32_t*, const int32_t*,
    const float*,
    float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_voxelize_to_voxels_bwd<double>(
    const double*, const double*, const double*,
    const int32_t*, const int32_t*,
    const double*,
    double*, double*, double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_voxelize_to_voxels_bwd2<float>(
    const float*, const float*, const float*,
    const int32_t*, const int32_t*,
    const float*, const float*,
    float*, float*, float*,
    float*, float*, float*,
    float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_voxelize_to_voxels_bwd2<double>(
    const double*, const double*, const double*,
    const int32_t*, const int32_t*,
    const double*, const double*,
    double*, double*, double*,
    double*, double*, double*,
    double*, double*, double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

}  // namespace gsplat2ct
