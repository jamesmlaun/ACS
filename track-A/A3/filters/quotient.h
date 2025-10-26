#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <cstddef>
#include <algorithm>
#include "hash_utils.h"

// ------------------------------------------------------------
// Quotient Filter (seeded version) - corrected run maintenance
// ------------------------------------------------------------
class QuotientFilter {
public:
    QuotientFilter(size_t n, double fpr, uint64_t seed = 0);

    void insert(uint64_t key);
    bool contains(uint64_t key) const;
    bool remove(uint64_t key);
    void finalize() {}

    // Metrics (unchanged API)
    double avg_probe_length() const {
        return m_total_inserts ? (double)m_total_probes / (double)m_total_inserts : 0.0;
    }
    double avg_cluster_length() const {
        size_t total_len = 0, total_clusters = 0;
        for (size_t i = 1; i < m_cluster_hist.size(); ++i) {
            total_len += m_cluster_hist[i] * i;
            total_clusters += m_cluster_hist[i];
        }
        return total_clusters ? static_cast<double>(total_len) / total_clusters : 0.0;
    }
    size_t max_cluster_length() const { return m_max_cluster; }
    const std::vector<size_t>& cluster_histogram() const { return m_cluster_hist; }

    void reset_metrics() {
        m_total_probes = 0;
        m_total_inserts = 0;
        m_max_cluster = 0;
        std::fill(m_cluster_hist.begin(), m_cluster_hist.end(), 0);
    }
    void set_metrics_enabled(bool v) { m_metrics_enabled = v; }

    size_t capacity_slots() const { return capacity; }

    void clear() {
        for (auto &slot : table) slot = {0,0,0,0};
        std::fill(is_occupied.begin(), is_occupied.end(), 0);
        m_size = 0;
        reset_metrics();
    }

    size_t size() const { return m_size; }

    // Introspection
    int fp_bits() const { return static_cast<int>(rbits); }
    size_t table_bytes() const {
        // 1 remainder byte min; we store as 64-bit for simplicity here
        return table.size() * sizeof(Slot) + is_occupied.size(); // rough
    }
    size_t metadata_bytes() const { return is_occupied.size(); }

private:
    struct Slot {
        uint64_t remainder;   // we use only rbits low bits (nonzero)
        uint8_t  shifted;     // 1 if stored away from canonical bucket
        uint8_t  continuation;// 1 if not the first element in its run
        uint8_t  pad;         // keep 16-byte slot for alignment
    };

    std::vector<Slot> table;        // slots
    std::vector<uint8_t> is_occupied; // per-bucket "occupied" map
    size_t m_size = 0;

    size_t qbits = 0;     // quotient bits
    size_t rbits = 0;     // remainder bits
    size_t capacity = 0;  // slots (power of two)
    uint64_t seed_ = 0;   // hash seed
    std::mt19937_64 gen;

    // Metrics
    size_t m_total_probes = 0;
    size_t m_total_inserts = 0;
    size_t m_max_cluster = 0;
    std::vector<size_t> m_cluster_hist;
    bool m_metrics_enabled = true;

    // helpers
    inline size_t mod_index(size_t i) const { return i & (capacity - 1); }
    inline bool is_empty(size_t i) const {
        const auto &s = table[i];
        return (s.remainder | s.shifted | s.continuation) == 0;
    }

    // map key -> (q,r)
    inline void qr(uint64_t key, size_t &q, uint64_t &r) const {
        uint64_t h = hash64(key, seed_);
        q = (h >> rbits) & (capacity - 1);
        r = h & ((1ULL << rbits) - 1);
        if (r == 0) r = 1; // reserve 0 as empty sentinel
    }

    // cluster head that includes bucket q
    size_t cluster_head_for(size_t q) const;
    // start of run for bucket q (or insertion point if run not present)
    size_t run_start_for(size_t q) const;
    // end of run (returns last index of run given its start)
    size_t run_end_from(size_t run_start) const;

    // shift block [start..end] one step forward (toward +1), end is empty
    void shift_right(size_t start, size_t end);

    // metrics utilities
    void record_after_insert(size_t q, size_t probe_dist);
};
