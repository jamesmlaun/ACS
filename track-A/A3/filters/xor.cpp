#include "xor.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

// ------------------------------------------------------------
// Constructor — keep the same signature / fields you already use
// ------------------------------------------------------------
XorFilter::XorFilter(size_t n, double fpr, uint64_t seed)
    : capacity(0),
      fingerprint_bits(0),
      seed_(seed),
      seed1(0),
      seed2(0),
      seed3(0),
      fpsalt(0),
      built(false),
      fingerprints(),
      pending_keys(),
      gen(seed)
{
    // Bits per fingerprint from target FPR; clamp 8..16 (your project convention)
    fingerprint_bits = static_cast<uint8_t>(std::ceil(std::log2(1.0 / fpr)) + 1);
    fingerprint_bits = std::clamp<uint8_t>(fingerprint_bits, 6, 16);

    // Slightly larger overprovision improves peel success at N ~ 1e6
    double scale = std::clamp(std::log10(1.0 / fpr) / 3.0, 0.7, 2.0);
    capacity = static_cast<size_t>(std::ceil(n * (1.23 + 0.10 * scale)));
    if (capacity < 8) capacity = 8;

    fingerprints.assign(capacity, 0u);
    pending_keys.reserve(n);

    reseed();
}

// ------------------------------------------------------------
// Reseed salts for 3 indices + fingerprint
// ------------------------------------------------------------
void XorFilter::reseed() {
    seed1  = hash64(seed_ ^ 0x9e3779b97f4a7c15ULL, seed_);
    seed2  = hash64(seed_ ^ 0xbf58476d1ce4e5b9ULL, seed_);
    seed3  = hash64(seed_ ^ 0x94d049bb133111ebULL, seed_);
    fpsalt = hash64(seed_ ^ 0xF00DBABEDEADBEEFULL, seed_);
}

// ------------------------------------------------------------
// Fingerprint (we return uint64_t per header, but only low bits are used)
// ------------------------------------------------------------
uint64_t XorFilter::fingerprint(uint64_t key) const {
    const uint64_t ms = (1ULL << fingerprint_bits) - 1ULL;
    uint64_t fp = hash64(key, fpsalt) & ms;
    if (fp == 0) fp = 1; // avoid zero so XOR sums work nicely
    return fp;
}

// ------------------------------------------------------------
// Three independent index functions (i = 0,1,2)
// ------------------------------------------------------------
inline size_t XorFilter::hash_index(uint64_t key, int i) const {
    const uint64_t s = (i == 0) ? seed1 : (i == 1 ? seed2 : seed3);
    return static_cast<size_t>(hash64(key, s) % capacity);
}

// ------------------------------------------------------------
// O(N) peel using degree + XOR-of-edge-ids per vertex.
// Bounded reseed to avoid indefinite hang on unlucky seeds.
// ------------------------------------------------------------
void XorFilter::build(const std::vector<uint64_t>& keys) {
    const int MAX_ATTEMPTS = 6;   // a handful of independent tries
    bool success = false;

    for (int attempt = 0; attempt < MAX_ATTEMPTS && !success; ++attempt) {
        // Re-randomize the hashing seeds each attempt (independent random graph)
        seed_ ^= (0xA5A5A5A5A5A5A5A5ULL + attempt * 0x9E3779B97F4A7C15ULL);
        reseed();

        std::fill(fingerprints.begin(), fingerprints.end(), static_cast<uint16_t>(0));

        const size_t n = keys.size();
        // Precompute three positions per edge (key)
        std::vector<std::array<size_t,3>> pos(n);
        pos.reserve(n);

        // Per-vertex degree and XOR-of-edge-ids accumulator
        std::vector<uint32_t> deg(capacity, 0);
        std::vector<size_t>   edge_xor(capacity, 0);  // XOR of incident edge indices (size_t)

        for (size_t i = 0; i < n; ++i) {
            const uint64_t k = keys[i];
            size_t p0 = hash_index(k, 0);
            size_t p1 = hash_index(k, 1);
            size_t p2 = hash_index(k, 2);
            pos[i] = {p0, p1, p2};
            // Update per-vertex accumulators
            ++deg[p0]; edge_xor[p0] ^= i;
            ++deg[p1]; edge_xor[p1] ^= i;
            ++deg[p2]; edge_xor[p2] ^= i;
        }

        // Initialize stack of degree-1 vertices
        std::vector<size_t> stack;
        stack.reserve(n);
        for (size_t v = 0; v < capacity; ++v) {
            if (deg[v] == 1) stack.push_back(v);
        }

        // Peeling order: store (edge_id, slot_used)
        std::vector<std::pair<size_t,size_t>> order;
        order.reserve(n);

        // Peel: O(N)
        while (!stack.empty()) {
            const size_t v = stack.back();
            stack.pop_back();
            if (deg[v] != 1) continue;   // might have been updated already

            // The unique incident edge id at v is edge_xor[v]
            const size_t e = edge_xor[v];
            if (e >= n) continue;        // stale or invalid

            // Push this edge and the slot v we'll assign later
            order.emplace_back(e, v);

            // "Remove" this edge from the graph: for its three vertices,
            // decrement degree and XOR out the edge id.
            const auto &pp = pos[e];
            for (int j = 0; j < 3; ++j) {
                const size_t u = pp[j];
                if (deg[u] > 0) {
                    deg[u] -= 1;
                    edge_xor[u] ^= e;
                    if (deg[u] == 1) stack.push_back(u);
                }
            }
            // Mark this edge as processed by moving its pos to sentinel
            // (not strictly necessary with XOR-of-ids trick, but harmless)
            pos[e][0] = pos[e][1] = pos[e][2] = capacity;
        }

        if (order.size() != n) {
            std::cerr << "[WARN] XOR peel attempt " << (attempt + 1)
                      << " failed (" << order.size() << "/" << n
                      << " peeled). Retrying with new seeds...\n";
            continue; // try another random graph
        }

        // Reverse-assign fingerprints following recorded order
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const size_t e    = it->first;   // edge id (key index)
            const size_t slot = it->second;  // vertex (table slot) chosen for this key
            const uint64_t key = keys[e];

            const uint16_t fp = static_cast<uint16_t>(fingerprint(key)) & fp_mask();
            uint16_t acc = fp;

            const size_t p0 = hash_index(key, 0);
            const size_t p1 = hash_index(key, 1);
            const size_t p2 = hash_index(key, 2);

            if (p0 != slot) acc ^= fingerprints[p0];
            if (p1 != slot) acc ^= fingerprints[p1];
            if (p2 != slot) acc ^= fingerprints[p2];

            fingerprints[slot] = static_cast<uint16_t>(acc & fp_mask());
        }

        built = true;
        success = true;
        if (attempt > 0) {
            std::cerr << "[INFO] XOR build succeeded on retry " << (attempt + 1)
                      << " (N=" << keys.size() << ", cap=" << capacity << ")\n";
        }
    }

    if (!success) {
        std::cerr << "[ERROR] XOR build failed after " << MAX_ATTEMPTS
                  << " attempts (N=" << keys.size() << ", cap=" << capacity << ")\n";
        built = false;
        // You can throw here if you want to abort the whole run:
        // throw std::runtime_error("XOR build failed");
    }
}

// ------------------------------------------------------------
// Query
// ------------------------------------------------------------
bool XorFilter::contains(uint64_t key) const {
    if (!built) return false;

    const uint16_t fp = static_cast<uint16_t>(fingerprint(key)) & fp_mask();
    uint16_t acc = 0;
    acc ^= fingerprints[hash_index(key, 0)];
    acc ^= fingerprints[hash_index(key, 1)];
    acc ^= fingerprints[hash_index(key, 2)];
    acc &= fp_mask();

    return acc == fp;
}

// ------------------------------------------------------------
// Static build API (insert during build, then finalize once)
// ------------------------------------------------------------
void XorFilter::insert(uint64_t key) {
    if (built) return;
    pending_keys.push_back(key);
}

void XorFilter::finalize() {
    if (!built && !pending_keys.empty()) {
        build(pending_keys);
        pending_keys.clear();
    }
}
