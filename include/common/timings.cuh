#pragma once

#ifdef PROFILE_TIMINGS

#include <cstdint>
#include <cuda_runtime.h>

namespace timing_profile {

// Keep this record exactly 16 bytes so each event is committed with one
// vectorized global-memory store.
struct alignas(16) TimingRecord {
    uint64_t timestamp;
    uint32_t event_id;
    uint32_t payload;
};
static_assert(sizeof(TimingRecord) == 16);

inline constexpr uint32_t EVENTS_PER_BLOCK = 65536;

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t timestamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
    return timestamp;
}

__device__ __forceinline__ void store_record(TimingRecord* records,
                                              uint64_t index,
                                              uint64_t timestamp,
                                              uint32_t event_id,
                                              uint32_t payload) {
    ulonglong2 record;
    record.x = timestamp;
    record.y = (static_cast<uint64_t>(event_id) << 32) | payload;
    reinterpret_cast<ulonglong2*>(records)[index] = record;
}

__device__ __forceinline__ void emit_at(TimingRecord* records,
                                        uint32_t* shared_head,
                                        uint64_t timestamp,
                                        uint32_t event_id,
                                        uint32_t payload) {
    const uint32_t index = atomicAdd(shared_head, 1u);
    if (records != nullptr && index < EVENTS_PER_BLOCK) {
        store_record(records,
                     static_cast<uint64_t>(blockIdx.x) * EVENTS_PER_BLOCK + index,
                     timestamp,
                     event_id,
                     payload);
    }
}

__device__ __forceinline__ void emit(TimingRecord* records,
                                     uint32_t* shared_head,
                                     uint32_t event_id,
                                     uint32_t payload) {
    emit_at(records, shared_head, globaltimer_ns(), event_id, payload);
}

}  // namespace timing_profile

#endif  // PROFILE_TIMINGS
