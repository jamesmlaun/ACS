#include "quotient.h"
#include "hash_utils.h"
#include <cmath>
#include <iostream>
#include <algorithm>

// ------------------------------------------------------------
// Corrected Quotient Filter
// ------------------------------------------------------------
QuotientFilter::QuotientFilter(size_t n, double fpr, uint64_t seed)
    : seed_(seed), gen(seed)
{
    // Choose remainder bits from fpr (6..16)
    size_t want_r = static_cast<size_t>(std::ceil(std::log2(1.0 / fpr))) + 1;
    rbits = std::clamp<size_t>(want_r, 6, 16);

    // Capacity with breathing room (QF needs slack for runs)
    double target_load = 0.70; // safe default
    size_t base = static_cast<size_t>(std::ceil(double(n) / target_load));
    qbits = static_cast<size_t>(std::ceil(std::log2(std::max<size_t>(1, base))));
    capacity = 1ULL << qbits;

    table.assign(capacity, Slot{0,0,0,0});
    is_occupied.assign(capacity, 0);

    m_size = 0;
    m_total_probes = 0;
    m_total_inserts = 0;
    m_max_cluster = 0;
    m_cluster_hist.assign(65, 0);
    m_metrics_enabled = true;
}

// ---- cluster/run navigation -------------------------------------------------

size_t QuotientFilter::cluster_head_for(size_t q) const {
    // Walk backwards while current slot is shifted => still within same cluster
    size_t pos = q;
    size_t guard = 0;
    while (table[pos].shifted && guard++ < capacity)
        pos = mod_index(pos - 1);
    return pos;
}

size_t QuotientFilter::run_start_for(size_t q) const {
    // Find cluster head and its bucket index
    size_t head = cluster_head_for(q);

    // Determine bucket index at cluster head:
    // If head == q, head_bucket is q; otherwise, head_bucket = q - distance(head..q)
    size_t dist = (q >= head) ? (q - head) : (capacity - (head - q));
    size_t head_bucket = (q + capacity - dist) & (capacity - 1);

    // Walk runs from head_bucket up to q, advancing 'pos' accordingly
    size_t pos = head;
    size_t b = head_bucket;
    while (b != q) {
        if (is_occupied[b]) {
            // skip this run: first element at pos, then all continuation=1
            pos = mod_index(pos + 1);
            while (table[pos].continuation)
                pos = mod_index(pos + 1);
        }
        b = (b + 1) & (capacity - 1);
    }
    // At this point, pos is at the start of run for q if it exists,
    // or the insertion point for a new run for q if !is_occupied[q].
    return pos;
}

size_t QuotientFilter::run_end_from(size_t run_start) const {
    size_t pos = run_start;
    // move while next belongs to same run (continuation=1)
    while (table[mod_index(pos + 1)].continuation)
        pos = mod_index(pos + 1);
    return pos;
}

// ---- shifting ---------------------------------------------------------------

void QuotientFilter::shift_right(size_t start, size_t end_empty) {
    // Shift block [start .. end_empty-1] forward by one into end_empty.
    // Precondition: end_empty is empty.
    size_t i = end_empty;
    while (i != start) {
        size_t prev = mod_index(i - 1);
        table[i] = table[prev];
        table[i].shifted = 1; // moved => definitely shifted
        i = prev;
    }
    // 'start' is now free for caller to fill
}

// ---- contains ---------------------------------------------------------------

bool QuotientFilter::contains(uint64_t key) const {
    size_t q; uint64_t r;
    qr(key, q, r);

    if (!is_occupied[q]) return false;

    // locate run start for q, then scan that run
    size_t run_start = run_start_for(q);
    size_t pos = run_start;

    // scan forward until run ends
    while (true) {
        const auto &s = table[pos];
        if ((s.shifted || s.continuation || (pos == q && is_occupied[q])) && s.remainder == r)
            return true;

        // stop if next is not continuation => run ends
        size_t next = mod_index(pos + 1);
        if (!table[next].continuation)
            break;
        pos = next;
    }
    return false;
}

// ---- insert -----------------------------------------------------------------

void QuotientFilter::insert(uint64_t key) {
    size_t q; uint64_t r;
    qr(key, q, r);

    // Mark bucket as having a run
    is_occupied[q] = 1;

    // Find where to place: end of existing run, or create a new run
    size_t run_start = run_start_for(q);
    size_t insert_pos;
    if (table[run_start].continuation == 0 && (table[run_start].shifted || run_start != q) && is_occupied[q]==1 && !is_empty(run_start)) {
        // a run exists; append at its end
        insert_pos = mod_index(run_end_from(run_start) + 1);
    } else if (!is_empty(run_start) && is_occupied[q]) {
        // run exists but run_start is occupied by some element of it
        insert_pos = mod_index(run_end_from(run_start) + 1);
    } else {
        // new run: insert at run_start
        insert_pos = run_start;
    }

    // Find first empty slot at/after insert_pos
    size_t pos = insert_pos;
    size_t probe = 0;
    while (!is_empty(pos)) {
        pos = mod_index(pos + 1);
        if (++probe >= capacity) {
            std::cerr << "[ERROR] QuotientFilter::insert - table full\n";
            return;
        }
    }

    // Shift block forward by one to make room
    if (pos != insert_pos)
        shift_right(insert_pos, pos);

    // Place the new element
    Slot s{};
    s.remainder = r;
    s.shifted = (insert_pos != q);
    // It's a continuation iff we are not the first element of the run
    s.continuation = (insert_pos != run_start) ? 1 : 0;
    table[insert_pos] = s;

    ++m_size;

    // --- optional: quick local normalization at run boundary ---
    // Ensure the element at run_start is always flagged as continuation=0
    table[run_start].continuation = 0;

    // Metrics (bounded)
    if (m_metrics_enabled) {
        record_after_insert(q, probe);
    }
}

// ---- remove -----------------------------------------------------------------

bool QuotientFilter::remove(uint64_t key) {
    size_t q; uint64_t r;
    qr(key, q, r);

    if (!is_occupied[q]) return false;

    size_t run_start = run_start_for(q);
    size_t pos = run_start;

    // Find the element inside the run
    while (true) {
        auto &s = table[pos];
        if (!is_empty(pos) && s.remainder == r) {
            // Remove by clearing slot; NOTE: we keep simple clear (no backward-shift compaction)
            table[pos] = Slot{0,0,0,0};
            if (m_size) --m_size;

            // If the run becomes empty, clear occupied bit
            // (detect run empty by checking next slot not continuation and current cleared)
            if (!table[run_start].continuation && is_empty(run_start)) {
                is_occupied[q] = 0;
            }
            return true;
        }
        size_t next = mod_index(pos + 1);
        if (!table[next].continuation) break; // end of run
        pos = next;
    }
    return false;
}

// ---- metrics ----------------------------------------------------------------

void QuotientFilter::record_after_insert(size_t q, size_t probe_dist) {
    m_total_probes += probe_dist;
    ++m_total_inserts;

    // cluster length: from cluster head through contiguous (shifted/continuation or head)
    size_t head = cluster_head_for(q);
    size_t len = 1;
    size_t pos = head;
    while (true) {
        size_t next = mod_index(pos + 1);
        if (is_empty(next) || (!table[next].shifted && !table[next].continuation)) break;
        ++len;
        pos = next;
        if (len > capacity) break;
    }
    m_max_cluster = std::max(m_max_cluster, len);
    size_t bin = (len <= 63) ? len : 64;
    if (bin < m_cluster_hist.size()) ++m_cluster_hist[bin];
}
