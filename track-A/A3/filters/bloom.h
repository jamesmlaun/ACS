#pragma once
#include <vector>
#include <cstdint>
#include "hash_utils.h"

class BloomFilter {
public:
    BloomFilter(size_t n, double fpr, uint64_t seed = 0);

    void insert(uint64_t key);
    bool contains(uint64_t key) const;
    void finalize() {}

    size_t bits() const { return m_bits; }

    // Introspection helpers
    int fp_bits() const { return -1; } // Bloom filters store bits, not per-key fingerprints
    size_t table_bytes() const { return (m_bits + 7) / 8; }
    size_t metadata_bytes() const {  return sizeof(size_t) * 2 + sizeof(uint64_t); }

private:
    size_t m_bits;                  // total number of bits
    int k_hashes;                   // number of hash functions
    std::vector<uint64_t> bitset;   // bit array
    uint64_t seed_;                 // base seed for hashing
};
