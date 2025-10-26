#include "cuckoo.h"
#include <cmath>
#include <iostream>
#include <algorithm>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
CuckooFilter::CuckooFilter(size_t n, double fpr, uint64_t seed)
    : seed_(seed), gen(seed)
{
    // Dynamic fingerprint width (8–16 bits)
    // Heuristic: based on target fpr and bucketization
    double desired = std::ceil(std::log2(2.0 * BUCKET_SIZE / fpr));
    fingerprint_bits = static_cast<uint8_t>(desired);
    fingerprint_bits = std::clamp<uint8_t>(fingerprint_bits, 8, 16);

    // Higher accuracy (smaller FPR) ⇒ lower load factor ⇒ more buckets
    double base_load = 0.70;          // ~30% headroom
    double scale = std::clamp( std::log10(1.0 / fpr) / 3.0, 0.5, 1.5 );
    double target_load = std::max(0.35, base_load / (scale * 1.2));

    // Use ~70% target load factor to reduce evictions (same style as original)
    num_buckets = static_cast<size_t>(
        std::ceil((n / (double)BUCKET_SIZE) / target_load)
    );
    if (num_buckets < 1) num_buckets = 1;

    table.resize(num_buckets);
    for (auto &bucket : table)
        bucket.reserve(BUCKET_SIZE);

    stash.reserve(STASH_SIZE);

    // Metrics start at zero for each filter instance
    m_eviction_attempts = 0;
    m_stash_hits = 0;
    m_failed_inserts = 0;
}

// ------------------------------------------------------------
// Fingerprint generation (8–16 bits) using seeded xxHash
// ------------------------------------------------------------
uint16_t CuckooFilter::fingerprint(uint64_t key) const {
    // Use a distinct seed for fingerprints (derived from instance seed_)
    uint64_t h = hash64(key, seed_ ^ 0xA5A5A5A5A5A5A5A5ULL);
    uint64_t mask = (1ULL << fingerprint_bits) - 1ULL;
    uint16_t fp = static_cast<uint16_t>(h & mask);
    if (fp == 0) fp = 1; // avoid zero fingerprint (reserved)
    return fp;
}

// ------------------------------------------------------------
// Primary bucket index using seeded hash
// ------------------------------------------------------------
size_t CuckooFilter::index1(uint64_t key) const {
    uint64_t h = hash64(key, seed_);
    return static_cast<size_t>(h % num_buckets);
}

// ------------------------------------------------------------
// Secondary bucket index (independent mixing, using fp & seed)
// ------------------------------------------------------------
size_t CuckooFilter::index2(size_t i1, uint16_t fp) const {
    // Derive independent hash from the fingerprint
    uint64_t h = hash64(fp, seed_ ^ 0x9e3779b97f4a7c15ULL);
    return (i1 ^ (h % num_buckets)) % num_buckets;
}

// ------------------------------------------------------------
// Insert with adaptive eviction & stash
// ------------------------------------------------------------
void CuckooFilter::insert(uint64_t key) {
    uint16_t fp = fingerprint(key);
    size_t i1 = index1(key);
    size_t i2 = index2(i1, fp);

    // Try primary / alternate buckets
    if (table[i1].size() < BUCKET_SIZE) { table[i1].push_back(fp); ++m_items; return; }
    if (table[i2].size() < BUCKET_SIZE) { table[i2].push_back(fp); ++m_items; return; }

    // Adaptive MAX_KICKS (bounded)
    int max_kicks = std::max(1000, static_cast<int>(10 * std::log2(std::max<size_t>(2, num_buckets))));

    // Eviction loop
    size_t i = (gen() % 2 == 0) ? i1 : i2;
    uint16_t cur_fp = fp;

    for (int kick = 0; kick < max_kicks*2; ++kick) {
        ++m_eviction_attempts;
        int slot = static_cast<int>(gen() % table[i].size());
        std::swap(cur_fp, table[i][slot]);

        // compute alternate bucket of the evicted fingerprint
        size_t alt = (i ^ (hash64(cur_fp, seed_ ^ 0x9e3779b97f4a7c15ULL) % num_buckets)) % num_buckets;

        if (table[alt].size() < BUCKET_SIZE) {
            table[alt].push_back(cur_fp);
            ++m_items;
            return;
        }
        i = alt;  // continue eviction chain
    }

    if (stash.size() < STASH_SIZE) {
        stash.push_back(cur_fp);
        ++m_stash_hits;
        ++m_items;
    } else {
        // Try to keep it somewhere legal instead of dropping
        size_t i1r = index1(cur_fp);
        size_t i2r = index2(i1r, cur_fp);
        size_t t1 = table[i1r].size();
        size_t t2 = table[i2r].size();
        if (t1 < BUCKET_SIZE) {
            table[i1r].push_back(cur_fp);
        } else if (t2 < BUCKET_SIZE) {
            table[i2r].push_back(cur_fp);
        } else {
            ++m_failed_inserts;
            std::fprintf(stderr, "[WARN] CuckooFilter: failed insert after stash full\n");
        }
    }
}

// ------------------------------------------------------------
// Membership query
// ------------------------------------------------------------
bool CuckooFilter::contains(uint64_t key) const {
    const uint16_t fp = fingerprint(key);
    const size_t i1 = index1(key);
    const size_t i2 = index2(i1, fp);

    // Check primary bucket first
    const auto &b1 = table[i1];
    for (auto v : b1) {
        if (v == fp) return true; // early return
    }

    // Check alternate bucket only if necessary
    const auto &b2 = table[i2];
    for (auto v : b2) {
        if (v == fp) return true;
    }

    // Finally check stash (rare)
    for (auto v : stash) {
        if (v == fp) return true;
    }

    return false; // miss
}

// ------------------------------------------------------------
// Delete fingerprint from table & stash
// ------------------------------------------------------------
bool CuckooFilter::remove(uint64_t key) {
    uint16_t fp = fingerprint(key);
    size_t i1 = index1(key);
    size_t i2 = index2(i1, fp);

    bool removed = false;

    auto erase_from = [&](size_t idx) -> size_t {
        auto &b = table[idx];
        size_t cnt = 0;
        for (auto it = b.begin(); it != b.end();) {
            if (*it == fp) {
                it = b.erase(it);
                ++cnt;
            } else ++it;
        }
        return cnt;
    };

    size_t c1 = erase_from(i1); if (c1) removed = true, m_items -= c1;
    size_t c2 = erase_from(i2); if (c2) removed = true, m_items -= c2;

    // Remove from stash only if not found elsewhere
    if (!removed) {
        auto it = std::find(stash.begin(), stash.end(), fp);
        if (it != stash.end()) {
            stash.erase(it);
            removed = true;
            if (m_items) --m_items;
        }
    }

    return removed;
}
