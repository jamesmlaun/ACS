// ============================================================
// test_filters.cpp — unified correctness tests for A3 filters
// ============================================================

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <algorithm>
#include <cassert>
#include <chrono>

#include "../filters/bloom.h"
#include "../filters/cuckoo.h"
#include "../filters/xor.h"
#include "../filters/quotient.h"

// ------------------------------------------------------------
// Shared base test for insert/query correctness
// ------------------------------------------------------------
template <typename FilterT>
bool base_filter_test(FilterT& filter, const std::string& name,
                      size_t N, double fpr_target)
{
    std::cout << "=== Testing " << name << " ===\n";

    std::mt19937_64 gen(42);
    std::uniform_int_distribution<uint64_t> dist;

    std::vector<uint64_t> keys(N);
    for (auto& k : keys) k = dist(gen);

    // Insert half the keys
    for (size_t i = 0; i < N / 2; ++i)
        filter.insert(keys[i]);
    filter.finalize();

    // Measure base rates
    size_t pos_hits = 0, false_pos = 0;
    for (size_t i = 0; i < N / 2; ++i)
        if (filter.contains(keys[i])) pos_hits++;
    for (size_t i = N / 2; i < N; ++i)
        if (filter.contains(keys[i])) false_pos++;

    double tp_rate = double(pos_hits) / (N / 2);
    double fp_rate = double(false_pos) / (N / 2);

    std::cout << "  True positives: " << pos_hits << "/" << (N / 2)
              << " (" << tp_rate * 100.0 << "%)\n";
    std::cout << "  False positives: " << false_pos << "/" << (N / 2)
              << " (" << fp_rate * 100.0 << "%, target "
              << fpr_target * 100.0 << "%)\n";

    bool ok = (tp_rate > 0.999) && (fp_rate < fpr_target * 2.0);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " Base correctness test.\n\n";
    return ok;
}

// ------------------------------------------------------------
// Bloom filter test (baseline)
// ------------------------------------------------------------
bool test_bloom(size_t N, double fpr, uint64_t seed)
{
    BloomFilter bloom(N, fpr, seed);
    return base_filter_test(bloom, "BloomFilter", N, fpr);
}

// ------------------------------------------------------------
// Cuckoo filter extended test (insert/delete)
// ------------------------------------------------------------
bool test_cuckoo(size_t N, double fpr, uint64_t seed)
{
    CuckooFilter cf(N, fpr, seed);
    bool ok1 = base_filter_test(cf, "CuckooFilter", N, fpr);

    std::mt19937_64 gen(seed ^ 0xC00C00ULL);
    std::uniform_int_distribution<uint64_t> dist;

    std::vector<uint64_t> keys(N);
    for (auto& k : keys) k = dist(gen);

    // Insert N/2 keys
    for (size_t i = 0; i < N / 2; ++i)
        cf.insert(keys[i]);

    // Delete half of the inserted keys
    for (size_t i = 0; i < N / 4; ++i)
        cf.remove(keys[i]);

    // Verify deleted elements are gone
    size_t residual = 0;
    for (size_t i = 0; i < N / 4; ++i)
        if (cf.contains(keys[i])) residual++;

    double residual_rate = double(residual) / (N / 4);

    std::cout << "  [Delete Test] Residual after delete: "
              << residual_rate * 100.0 << "%\n";

    bool ok2 = (residual_rate < 0.05);

    bool passed = ok1 && ok2;
    std::cout << (passed ? "[PASS]" : "[FAIL]")
              << " CuckooFilter full test (insert/delete/evict/stash)\n\n";
    return passed;
}

// ------------------------------------------------------------
// XOR filter test (read-only build/query correctness)
// ------------------------------------------------------------
bool test_xor(size_t N, double fpr, uint64_t seed)
{
    std::cout << "=== Testing XorFilter ===\n";
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<uint64_t> dist;

    std::vector<uint64_t> keys(N);
    for (auto& k : keys) k = dist(gen);

    XorFilter xf(N, fpr, seed);
    xf.build(keys);
    xf.finalize();

    size_t pos_hits = 0, false_pos = 0;
    for (size_t i = 0; i < N; ++i)
        if (xf.contains(keys[i])) pos_hits++;

    for (size_t i = 0; i < N; ++i) {
        uint64_t k2 = dist(gen);
        if (xf.contains(k2)) false_pos++;
    }

    double tp_rate = double(pos_hits) / N;
    double fp_rate = double(false_pos) / N;

    std::cout << "  True positives: " << pos_hits << "/" << N
              << " (" << tp_rate * 100.0 << "%)\n";
    std::cout << "  False positives: " << false_pos << "/" << N
              << " (" << fp_rate * 100.0 << "%, target "
              << fpr * 100.0 << "%)\n";

    bool ok = (tp_rate > 0.99) && (fp_rate < fpr * 2.0);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " XorFilter build/query test.\n\n";
    return ok;
}

// ------------------------------------------------------------
// Quotient filter correctness test (insert/delete)
// ------------------------------------------------------------
bool test_quotient(size_t N, double fpr, uint64_t seed)
{
    std::cout << "=== Testing QuotientFilter ===\n";
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<uint64_t> dist;

    std::vector<uint64_t> keys(N);
    for (auto& k : keys) k = dist(gen);

    QuotientFilter qf(N, fpr, seed);

    for (size_t i = 0; i < N / 2; ++i)
        qf.insert(keys[i]);
    qf.finalize();

    size_t tp = 0;
    for (size_t i = 0; i < N / 2; ++i)
        if (qf.contains(keys[i])) tp++;

    size_t fp = 0;
    for (size_t i = N / 2; i < N; ++i)
        if (qf.contains(keys[i])) fp++;

    double tp_rate = double(tp) / (N / 2);
    double fp_rate = double(fp) / (N / 2);

    std::cout << "  True positives: " << tp << "/" << (N / 2)
              << " (" << tp_rate * 100.0 << "%)\n";
    std::cout << "  False positives: " << fp << "/" << (N / 2)
              << " (" << fp_rate * 100.0 << "%, target "
              << fpr * 100.0 << "%)\n";

    bool ok1 = (tp_rate > 0.99) && (fp_rate < fpr * 2.0);
    std::cout << (ok1 ? "[PASS]" : "[FAIL]") << " Base correctness test.\n";

    // Delete subset and check residuals
    for (size_t i = 0; i < N / 4; ++i)
        qf.remove(keys[i]);

    size_t residual = 0;
    for (size_t i = 0; i < N / 4; ++i)
        if (qf.contains(keys[i])) residual++;

    double residual_rate = double(residual) / (N / 4);

    std::cout << "\n  [Delete Test] Residual after delete: "
              << residual_rate * 100.0 << "%\n";

    bool ok2 = (residual_rate < 0.05);

    bool passed = ok1 && ok2;
    std::cout << (passed ? "[PASS]" : "[FAIL]")
              << " QuotientFilter full test (insert/delete/metadata)\n\n";

    return passed;
}

// ------------------------------------------------------------
// CLI dispatcher
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string filter = "bloom";
    size_t N = 100000;
    double fpr = 0.01;
    uint64_t seed = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--filter=", 0) == 0)
            filter = arg.substr(9);
        else if (arg.rfind("--N=", 0) == 0)
            N = std::stoull(arg.substr(4));
        else if (arg.rfind("--fpr=", 0) == 0)
            fpr = std::stod(arg.substr(6));
        else if (arg.rfind("--seed=", 0) == 0)
            seed = std::stoull(arg.substr(7));
    }

    if (seed == 0)
        seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::cout << "[INFO] Running correctness test for '" << filter
              << "' with seed=" << seed << "\n";

    bool ok = false;
    if (filter == "bloom")
        ok = test_bloom(N, fpr, seed);
    else if (filter == "cuckoo")
        ok = test_cuckoo(N, fpr, seed);
    else if (filter == "xor")
        ok = test_xor(N, fpr, seed);
    else if (filter == "quotient")
        ok = test_quotient(N, fpr, seed);
    else {
        std::cerr << "Unknown filter: " << filter << "\n";
        return 1;
    }

    return ok ? 0 : 1;
}
