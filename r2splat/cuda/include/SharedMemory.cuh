#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace r2splat {

// Pad the int32 ID prefix before storing S scalars or vec<S> arrays.
// For double and an odd batch size, four padding bytes are required.
// Use this on both the host (allocation) and device (pointer offsets).
template <typename S>
__host__ __device__ constexpr uint32_t sampling_shared_ids_bytes(uint32_t count) {
    return (count * sizeof(int32_t) + alignof(S) - 1) / alignof(S) * alignof(S);
}

}  // namespace r2splat
