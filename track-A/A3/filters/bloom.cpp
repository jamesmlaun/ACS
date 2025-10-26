#include "bloom.h"
#include <cmath>
#include <iostream>

// ------------------------------------------------------------
// Bloom Filter Constructor
// ------------------------------------------------------------
BloomFilter::BloomFilter(size_t n, double fpr, uint64_t seed)
    : seed_(seed)
{
    double ln2 = std::log(2.0);
    m_bits = static_cast<size_t>(std::ceil(-(n * std::log(fpr)) / (ln2 * ln2)));
    k_hashes = std::max(1, static_cast<int>(std::round((m_bits / (double)n) * ln2)));

    // Round up to multiple of 64 bits for natural alignment
    size_t n_words = (m_bits + 63) / 64;

    // --- Patch: stable, pre-touched bitset allocation ---
    bitset.clear();
    bitset.resize(n_words, 0ULL);

    // Touch each cache line once to fault pages in and stabilize timing
    volatile uint64_t sink = 0;
    for (size_t i = 0; i < n_words; i += 8)  // touch roughly every 64 bytes
        sink ^= bitset[i];
    (void)sink;
}

// ------------------------------------------------------------
// Insert key into Bloom Filter using seeded xxHash
// ------------------------------------------------------------
void BloomFilter::insert(uint64_t key) {
    // Kirsch–Mitzenmacher double hashing: k indices from two hashes
    const uint64_t h1 = hash64(key, seed_);
    // derive an independent second hash (force odd to avoid short cycles)
    const uint64_t h2 = (hash64(key, seed_ ^ 0x9e3779b97f4a7c15ULL) | 1ULL);

    for (int i = 0; i < k_hashes; ++i) {
        size_t pos = static_cast<size_t>((h1 + (uint64_t)i * h2) % m_bits);
        bitset[pos / 64] |= (1ULL << (pos % 64));
    }
}

// ------------------------------------------------------------
// Check membership
// ------------------------------------------------------------
bool BloomFilter::contains(uint64_t key) const {
    // Kirsch–Mitzenmacher double hashing
    const uint64_t h1 = hash64(key, seed_);
    const uint64_t h2 = (hash64(key, seed_ ^ 0x9e3779b97f4a7c15ULL) | 1ULL);

    for (int i = 0; i < k_hashes; ++i) {
        size_t pos = static_cast<size_t>((h1 + (uint64_t)i * h2) % m_bits);
        // immediate return if any bit missing
        if ((bitset[pos / 64] & (1ULL << (pos % 64))) == 0ULL)
            return false;
    }
    return true;
}