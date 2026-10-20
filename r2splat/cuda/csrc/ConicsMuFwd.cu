/**
 * @file ConicsMuFwd.cu
 * @brief Forward CUDA kernel: compute conics and mu from projected 3x3 covariance
 *        with blur (eps * I) applied before inversion.
 * 
 * Given the projected 3x3 covariance matrix tilde{Sigma} stored as 6 upper-triangular
 * elements (a, b, c, d, e, f), computes:
 *   1. conics = (hat{Sigma} + eps*I_2)^{-1}, stored as (u11, u12, u22)
 *   2. mu = sqrt(2*pi * det(tilde{Sigma} + eps*I_3) / det(hat{Sigma} + eps*I_2))
 * 
 * The blur ensures numerical stability (prevents singular inversions) and provides
 * anti-aliasing for sub-voxel Gaussians (Mip-Splatting style).
 * 
 * Evaluate the determinant ratio as a Schur complement. This avoids absolute
 * determinant thresholds that incorrectly discard small, well-conditioned
 * covariances, and the backward pass reuses the saved conics and mu.
 * 
 * Each thread processes one Gaussian (element of the C*N batch).
 * 
 * Author: Dy164
 * Date: 2026-02-10
 */

#include "ConicsMu.h"
#include "../include/common.h"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace r2splat {

// ============================================================================
// Forward Kernel
// ============================================================================

template <typename S>
__global__ void conics_mu_fwd_kernel(
    const S* __restrict__ cov2d_3x3,   // [total, 6]  (a, b, c, d, e, f)
    const S eps,                       // blur epsilon
    S* __restrict__ conics,            // [total, 3]  (u11, u12, u22)
    S* __restrict__ mus,               // [total]
    uint32_t total
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Read the 6 independent elements of tilde{Sigma}
    const S* cov = cov2d_3x3 + idx * 6;
    const S a = cov[0];  // tilde{Sigma}_{00}
    const S b = cov[1];  // tilde{Sigma}_{01}
    const S c = cov[2];  // tilde{Sigma}_{02}
    const S d = cov[3];  // tilde{Sigma}_{11}
    const S e = cov[4];  // tilde{Sigma}_{12}
    const S f = cov[5];  // tilde{Sigma}_{22}

    // Apply blur: add eps to diagonal elements
    const S ab = a + eps;  // (a + eps)
    const S db = d + eps;  // (d + eps)
    const S fb = f + eps;  // (f + eps)
    // Off-diagonal elements remain: b, c, e

    // ----------------------------------------------------------------
    // Blurred 2x2 sub-block: hat{Sigma}_b = [[ab, b], [b, db]]
    // ----------------------------------------------------------------
    // Normalize before forming the determinant so its magnitude does not depend
    // on the physical units or the Gaussian scale. Positive definite inputs
    // remain valid even when their unscaled determinant would underflow.
    const S scale = fmax(fmax(fabs(ab), fabs(db)), fabs(b));
    const S an = ab / scale;
    const S bn = b / scale;
    const S dn = db / scale;
    const S determinant = an * dn - bn * bn;
    const S u11 = (dn / determinant) / scale;
    const S u12 = (-bn / determinant) / scale;
    const S u22 = (an / determinant) / scale;
    if (!(ab > static_cast<S>(0) && db > static_cast<S>(0)
          && determinant > static_cast<S>(0))
        || !isfinite(u11) || !isfinite(u12) || !isfinite(u22)) {
        // Degenerate 2D covariance, output zeros
        conics[idx * 3 + 0] = static_cast<S>(0);
        conics[idx * 3 + 1] = static_cast<S>(0);
        conics[idx * 3 + 2] = static_cast<S>(0);
        mus[idx] = static_cast<S>(0);
        return;
    }

    // ================================================================
    // Compute conics = hat{Sigma}_b^{-1}
    // ================================================================
    conics[idx * 3 + 0] = u11;
    conics[idx * 3 + 1] = u12;
    conics[idx * 3 + 2] = u22;

    // ================================================================
    // det(Sigma_b) / det(hat{Sigma}_b) = fb - [c,e] P [c,e]^T
    // ================================================================
    const S w0 = u11 * c + u12 * e;
    const S w1 = u12 * c + u22 * e;
    const S ratio = fb - c * w0 - e * w1;
    if (ratio > static_cast<S>(0) && isfinite(ratio)) {
        mus[idx] = sqrt(static_cast<S>(2.0 * CUDART_PI) * ratio);
    } else {
        mus[idx] = static_cast<S>(0);
    }
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_conics_mu_fwd(
    const S* __restrict__ cov2d_3x3,
    S eps,
    S* __restrict__ conics,
    S* __restrict__ mus,
    uint32_t total,
    cudaStream_t stream
) {
    if (total == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (total + threads - 1) / threads;

    conics_mu_fwd_kernel<S><<<blocks, threads, 0, stream>>>(
        cov2d_3x3, eps, conics, mus, total
    );
}

// Explicit template instantiations
template void launch_conics_mu_fwd<float>(
    const float*, float, float*, float*, uint32_t, cudaStream_t
);
template void launch_conics_mu_fwd<double>(
    const double*, double, double*, double*, uint32_t, cudaStream_t
);

} // namespace r2splat
