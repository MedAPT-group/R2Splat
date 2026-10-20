#pragma once

#include <cuda_runtime.h>

namespace r2splat {

// Invert a positive-definite covariance without a scale-dependent determinant
// cutoff. Normalizing first avoids under/overflow in products of small/large
// variances. Both mapping passes must use the same validity test.
template <typename S>
__device__ __forceinline__ bool invert_spd3x3(
    S a00, S a01, S a02, S a11, S a12, S a22,
    S& i00, S& i01, S& i02, S& i11, S& i12, S& i22
) {
    i00 = i01 = i02 = i11 = i12 = i22 = static_cast<S>(0);
    if (!isfinite(a00) || !isfinite(a01) || !isfinite(a02) ||
        !isfinite(a11) || !isfinite(a12) || !isfinite(a22) ||
        a00 <= static_cast<S>(0) || a11 <= static_cast<S>(0) ||
        a22 <= static_cast<S>(0)) {
        return false;
    }

    const S scale = max(a00, max(a11, a22));
    a00 /= scale;
    a01 /= scale;
    a02 /= scale;
    a11 /= scale;
    a12 /= scale;
    a22 /= scale;

    // A / scale = L L^T. Positive pivots distinguish covariances from singular
    // and indefinite matrices; their absolute physical size is irrelevant.
    const S l00 = sqrt(a00);
    const S l10 = a01 / l00;
    const S l20 = a02 / l00;
    const S pivot11 = a11 - l10 * l10;
    if (!(pivot11 > static_cast<S>(0))) return false;
    const S l11 = sqrt(pivot11);
    const S l21 = (a12 - l20 * l10) / l11;
    const S pivot22 = a22 - l20 * l20 - l21 * l21;
    if (!(pivot22 > static_cast<S>(0))) return false;
    const S l22 = sqrt(pivot22);

    // K = L^{-1}, then A^{-1} = K^T K / scale.
    const S k00 = static_cast<S>(1) / l00;
    const S k11 = static_cast<S>(1) / l11;
    const S k22 = static_cast<S>(1) / l22;
    const S k10 = -l10 * k00 * k11;
    const S k21 = -l21 * k11 * k22;
    const S k20 = -(l20 * k00 + l21 * k10) * k22;
    const S p00 = (k00 * k00 + k10 * k10 + k20 * k20) / scale;
    const S p01 = (k10 * k11 + k20 * k21) / scale;
    const S p02 = k20 * k22 / scale;
    const S p11 = (k11 * k11 + k21 * k21) / scale;
    const S p12 = k21 * k22 / scale;
    const S p22 = k22 * k22 / scale;
    if (!isfinite(p00) || !isfinite(p01) || !isfinite(p02) ||
        !isfinite(p11) || !isfinite(p12) || !isfinite(p22)) {
        return false;
    }
    i00 = p00;
    i01 = p01;
    i02 = p02;
    i11 = p11;
    i12 = p12;
    i22 = p22;
    return true;
}

} // namespace r2splat
