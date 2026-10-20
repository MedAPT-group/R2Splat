/** Shared device math for physical covariance parameters. */
#ifndef R2SPLAT_PARAMETER_TRANSFORM_CUH
#define R2SPLAT_PARAMETER_TRANSFORM_CUH

#include <cuda_runtime.h>

namespace r2splat {

// --use_fast_math normally approximates float division and flushes denormals.
// Exact, non-FTZ division keeps finite very large/small raw quaternions usable.
__device__ __forceinline__ float parameter_divide(float a, float b) {
    float result;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ __forceinline__ double parameter_divide(double a, double b) {
    return __ddiv_rn(a, b);
}

__device__ __forceinline__ float quaternion_max_abs(const float* q) {
    unsigned int magnitude_bits = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        // Positive IEEE-754 bit patterns have the same ordering as their values.
        // Integer comparison also preserves subnormal input magnitudes.
        magnitude_bits = max(magnitude_bits, __float_as_uint(q[i]) & 0x7fffffffU);
    }
    return __uint_as_float(magnitude_bits);
}

__device__ __forceinline__ double quaternion_max_abs(const double* q) {
    return fmax(fmax(fabs(q[0]), fabs(q[1])), fmax(fabs(q[2]), fabs(q[3])));
}

__device__ __forceinline__ float parameter_sqrt(float value) {
    return __fsqrt_rn(value);
}

__device__ __forceinline__ double parameter_sqrt(double value) {
    return __dsqrt_rn(value);
}

template <typename S>
__device__ __forceinline__ void normalize_parameter_quaternion(
    const S* raw, S* q, S& magnitude, S& scaled_norm
) {
    magnitude = quaternion_max_abs(raw);
    S norm_sq = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        q[i] = parameter_divide(raw[i], magnitude);
        norm_sq += q[i] * q[i];
    }
    scaled_norm = parameter_sqrt(norm_sq);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        q[i] = parameter_divide(q[i], scaled_norm);
    }
}

template <typename S>
__device__ __forceinline__ void parameter_rotation(const S* q, S R[3][3]) {
    const S w = q[0], x = q[1], y = q[2], z = q[3];
    R[0][0] = S(1) - S(2) * (y * y + z * z);
    R[0][1] = S(2) * (x * y - w * z);
    R[0][2] = S(2) * (x * z + w * y);
    R[1][0] = S(2) * (x * y + w * z);
    R[1][1] = S(1) - S(2) * (x * x + z * z);
    R[1][2] = S(2) * (y * z - w * x);
    R[2][0] = S(2) * (x * z - w * y);
    R[2][1] = S(2) * (y * z + w * x);
    R[2][2] = S(1) - S(2) * (x * x + y * y);
}

} // namespace r2splat
#endif // R2SPLAT_PARAMETER_TRANSFORM_CUH
