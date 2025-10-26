#pragma once
#include <cstdint>
#include "xxhash.h"

// Unified xxHash-based fast hash API for all filters
// Distinct seeds => independent hash functions per filter
inline uint64_t hash64(uint64_t key, uint64_t seed) {
    return XXH64(&key, sizeof(key), seed);
}

// Generate 3 independent hash values (for XOR/Bloom filters)
inline void multi_hash64(uint64_t key, uint64_t base_seed, uint64_t out[3]) {
    out[0] = hash64(key, base_seed);
    out[1] = hash64(key, base_seed + 0x9e3779b97f4a7c15ULL);
    out[2] = hash64(key, base_seed ^ 0xbf58476d1ce4e5b9ULL);
}
