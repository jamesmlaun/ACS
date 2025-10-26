#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <utility>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include "hash_utils.h"

// CuckooFilter: bucketized cuckoo filter with bounded evictions & small stash
// - Compatible API: CuckooFilter(size_t n, double fpr, uint64_t seed)
// - Methods: insert, contains, remove, finalize()
// - Uses seeded XXH64 via hash_utils.h for all hash/fingerprint computations.

class CuckooFilter {
public:
    // ------------------------------------------------------------
    // Constructor & core API
    // ------------------------------------------------------------
    CuckooFilter(size_t n, double fpr, uint64_t seed = 0);

    void insert(uint64_t key);
    bool contains(uint64_t key) const;
    bool remove(uint64_t key);
    void finalize() {}
    
    size_t eviction_attempts() const { return m_eviction_attempts; }
    size_t stash_hits() const { return m_stash_hits; }
    size_t failed_inserts() const { return m_failed_inserts; }

    // Reset between trials (bench calls this if it runs multiple sub-runs)
    void reset_metrics() {
        m_eviction_attempts = 0;
        m_stash_hits = 0;
        m_failed_inserts = 0;
    }

    size_t size() const { return m_items; }
    size_t capacity_items() const { return num_buckets * BUCKET_SIZE + STASH_SIZE; }

    // Introspection helpers
    int fp_bits() const { return static_cast<int>(fingerprint_bits); }
    size_t table_bytes() const {
        return static_cast<size_t>(
            num_buckets * BUCKET_SIZE * (fingerprint_bits / 8.0)
        );
    }
    size_t metadata_bytes() const { return num_buckets + (STASH_SIZE * (fingerprint_bits / 8.0)); }

private:
    static constexpr int BUCKET_SIZE = 4;
    static constexpr int MAX_KICKS   = 1000;
    static constexpr int STASH_SIZE  = 16;

    std::vector<std::vector<uint16_t>> table;  // buckets
    std::vector<uint16_t> stash;               // overflow stash
    size_t num_buckets;
    uint8_t fingerprint_bits;
    uint64_t seed_;                             // per-instance base seed
    std::mt19937_64 gen;

    size_t m_eviction_attempts = 0;
    size_t m_stash_hits = 0;
    size_t m_failed_inserts = 0;
    size_t m_items = 0;

    // Hashing & internal helpers (now use hash_utils / XXH64)
    uint16_t fingerprint(uint64_t key) const;
    size_t index1(uint64_t key) const;
    size_t index2(size_t i1, uint16_t fp) const;
};
