/**
 * Common macros and type definitions for CUDA extensions
 * 
 * Following gsplat's pattern with templates, __restrict__, and common macros.
 * 
 * Author: Dy164
 * Date: 2025-12-17
 */

#ifndef GSPLAT2CT_CUDA_COMMON_H
#define GSPLAT2CT_CUDA_COMMON_H

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

// ============================================================================
// Common Macros (following gsplat style)
// ============================================================================

#define GSPLAT2CT_N_THREADS 256

#define GSPLAT2CT_CHECK_CUDA(x)                                                \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define GSPLAT2CT_CHECK_CONTIGUOUS(x)                                          \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define GSPLAT2CT_CHECK_INPUT(x)                                               \
    GSPLAT2CT_CHECK_CUDA(x);                                                   \
    GSPLAT2CT_CHECK_CONTIGUOUS(x)
#define GSPLAT2CT_DEVICE_GUARD(_ten)                                           \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define GSPLAT2CT_PRAGMA_UNROLL _Pragma("unroll")

// ============================================================================
// Type Definitions
// ============================================================================

namespace gsplat2ct {

// Operation type mapping for mixed precision support
template <typename T> struct OpType {
    typedef T type;
};

#ifdef __CUDA_ARCH__
template <> struct OpType<__nv_bfloat16> {
    typedef float type;
};

template <> struct OpType<__half> {
    typedef float type;
};
#endif

template <> struct OpType<c10::Half> {
    typedef float type;
};

template <> struct OpType<c10::BFloat16> {
    typedef float type;
};

} // namespace gsplat2ct

#endif // GSPLAT2CT_CUDA_COMMON_H
