#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>
#include <random>
#include "hash_utils.h"

// Static XOR filter:
// • Build-time construction from a fixed key set (peeling).
// • Three independent hash positions.
// • Compact fingerprint array (8–16 bits per entry).
// • Seeded non-cryptographic hash (hash64 with distinct seeds).
class XorFilter {
public:
    XorFilter(size_t n, double fpr, uint64_t seed = 0);

    // Build filter from fixed key set
    void build(const std::vector<uint64_t>& keys);

    // Query membership
    bool contains(uint64_t key) const;

    // Benchmark interface compatibility
    void insert(uint64_t key);   // Stage keys before build
    void finalize();             // Trigger build from staged keys

    // Introspection helpers
    int fp_bits() const { return static_cast<int>(fingerprint_bits); }
    size_t table_bytes() const {
        return capacity * (fingerprint_bits / 8.0);
    }
    size_t metadata_bytes() const {  return 0; }

private:
    // === Core parameters ===
    size_t capacity;             // Number of slots
    uint8_t fingerprint_bits;    // Bits per fingerprint
    uint64_t seed_;              // Base seed
    uint64_t seed1, seed2, seed3;// Distinct index seeds
    uint64_t fpsalt;             // Fingerprint seed
    bool built;                  // True if constructed

    // === Data ===
    std::vector<uint16_t> fingerprints; // Fingerprint array
    std::vector<uint64_t> pending_keys;// Keys staged before build
    std::mt19937_64 gen;               // Random engine (optional)

    // === Internal helpers ===
    // Mask helper to keep XOR ops within fingerprint_bits
    inline uint16_t fp_mask() const {
        return (fingerprint_bits >= 16) ? 0xFFFFu : static_cast<uint16_t>((1u << fingerprint_bits) - 1u);
    }
    void reseed();                                    // Derive seeds from base seed
    uint64_t fingerprint(uint64_t key) const;         // Compute nonzero fingerprint
    inline size_t hash_index(uint64_t key, int i) const; // Compute i-th index
};
