// =============================================================
// bench.cpp — ACS Project A3 (single-configuration runner)
// =============================================================
// Modes (via --mode):
//   - space   : Space vs Accuracy (BPE incl. metadata vs achieved FPR)
//   - lookup  : Lookup throughput & latency tails vs negative share
//   - insert  : Insert/delete throughput (dynamic filters only)
//   - threads : Thread scaling (read-mostly & balanced)
//
// Filter APIs (from your uploaded headers):
//   insert(uint64_t), contains(uint64_t), finalize()
//   fp_bits(), table_bytes(), metadata_bytes()
//   CuckooFilter::remove(uint64_t)
//   QuotientFilter::remove(uint64_t)
//
// Compiles cleanly with C++17 and g++ 11+
// =============================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>   // <-- for ::strlen
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// -------------------------------------------------------------
// Filters
// -------------------------------------------------------------
#include "../filters/bloom.h"
#include "../filters/cuckoo.h"
#include "../filters/xor.h"
#include "../filters/quotient.h"

// ---- high-res timer helper (ns) ----
#if defined(__linux__)
#include <time.h>
static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}
#else
#include <chrono>
static inline uint64_t now_ns() {
    auto t = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}
#endif

// -------------------------------------------------------------
// Helpers
// -------------------------------------------------------------
static inline uint64_t rand64(std::mt19937_64 &g) {
    std::uniform_int_distribution<uint64_t> d;
    return d(g);
}
static inline uint64_t ms_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now() - t0)
        .count();
}
static inline uint64_t us_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - t0)
        .count();
}
static inline void quantiles(std::vector<uint64_t> &v, double &p50, double &p95, double &p99) {
    if (v.empty()) { p50 = p95 = p99 = 0; return; }
    std::sort(v.begin(), v.end());
    auto idx = [&](double q) {
        size_t i = static_cast<size_t>(std::round(q * (v.size() - 1)));
        if (i >= v.size()) i = v.size() - 1;
        return i;
    };
    p50 = v[idx(0.5)];
    p95 = v[idx(0.95)];
    p99 = v[idx(0.99)];
}

// -------------------------------------------------------------
// Args
// -------------------------------------------------------------
struct Args {
    std::string mode = "space";
    std::string filter = "bloom";
    double fpr = 0.01;
    uint64_t n = 1'000'000;
    double neg_ratio = 0.5;
    double load_factor = 0.5;
    int threads = 1;
    int reps = 1;
    std::string workload = "read_mostly";
    std::string out = "results.csv";
    uint64_t seed = 0;
    bool warm = true;
};

static Args parse_args(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto sub = [&](const char *pre) { return s.substr(::strlen(pre)); }; // <- ::strlen from <cstring>
        if (s.rfind("--mode=", 0) == 0)       a.mode = sub("--mode=");
        else if (s.rfind("--filter=", 0) == 0)     a.filter = sub("--filter=");
        else if (s.rfind("--fpr=", 0) == 0)        a.fpr = std::stod(sub("--fpr="));
        else if (s.rfind("--n=", 0) == 0)          a.n = std::stoull(sub("--n="));
        else if (s.rfind("--neg_ratio=", 0) == 0)  a.neg_ratio = std::stod(sub("--neg_ratio="));
        else if (s.rfind("--load_factor=", 0) == 0)a.load_factor = std::stod(sub("--load_factor="));
        else if (s.rfind("--threads=", 0) == 0)    a.threads = std::stoi(sub("--threads="));
        else if (s.rfind("--reps=", 0) == 0)       a.reps = std::stoi(sub("--reps="));
        else if (s.rfind("--workload=", 0) == 0)   a.workload = sub("--workload=");
        else if (s.rfind("--out=", 0) == 0)        a.out = sub("--out=");
        else if (s.rfind("--seed=", 0) == 0)       a.seed = std::stoull(sub("--seed="));
        else if (s.rfind("--warm=", 0) == 0) {
            std::string val = sub("--warm=");
            if (val == "0" || val == "false" || val == "False") a.warm = false;
            else a.warm = true;
        }
    }
    return a;
}

static void ensure_header(std::ofstream &csv, const std::string &hdr) {
    csv.seekp(0, std::ios::end);
    if (csv.tellp() == 0) csv << hdr << "\n";
}

// -------------------------------------------------------------
// generic achieved FPR
// -------------------------------------------------------------
template <typename F>
static double achieved_fpr(const F &f, const std::vector<uint64_t> &neg) {
    size_t fp = 0;
    for (auto k : neg)
        if (f.contains(k)) ++fp;
    return neg.empty() ? 0.0 : double(fp) / double(neg.size());
}

// =============================================================
// 1. Space vs Accuracy
// =============================================================
template <typename FilterT>
static void do_space(const Args &a, std::ofstream &csv) {
    std::mt19937_64 gen(a.seed ? a.seed : 1234);
    std::vector<uint64_t> pos(a.n), neg(a.n);
    for (auto &k : pos) k = rand64(gen);
    std::unordered_set<uint64_t> set(pos.begin(), pos.end());
    for (size_t i = 0; i < a.n; ++i) {
        uint64_t x; do { x = rand64(gen); } while (set.count(x));
        neg[i] = x;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    FilterT f(a.n, a.fpr, a.seed);
    for (auto k : pos) f.insert(k);
    f.finalize();
    uint64_t build_ms = ms_since(t0);

    double achieved = achieved_fpr(f, neg);
    double bpe = (f.table_bytes() + f.metadata_bytes()) * 8.0 / double(a.n);

    csv << a.filter << "," << a.n << "," << a.fpr << ","
        << achieved << "," << bpe << ","
        << f.fp_bits() << ","
        << f.table_bytes() << ","
        << f.metadata_bytes() << ","
        << build_ms << ","
        << a.seed << "\n";

    std::cout << "[SPACE] " << a.filter
              << " fpr=" << a.fpr
              << " achieved=" << achieved
              << " bpe=" << bpe << "\n";
}

// =============================================================
// 2. Lookup throughput & latency tails (final sanity-checked version)
// =============================================================
template <typename FilterT>
static void do_lookup(const Args &a, std::ofstream &csv) {
    using namespace std::chrono;

    std::mt19937_64 gen(a.seed ? a.seed : 5678);

    // ------------------------
    // Prepare positive/negative key sets
    // ------------------------
    std::vector<uint64_t> pos(a.n), neg(a.n);
    for (auto &k : pos) k = rand64(gen);

    std::unordered_set<uint64_t> set(pos.begin(), pos.end());
    for (size_t i = 0; i < a.n; ++i) {
        uint64_t x;
        do { x = rand64(gen); } while (set.count(x));
        neg[i] = x;
    }

    // ------------------------
    // Sanity: check overlap between pos/neg sets
    // ------------------------
    {
        size_t overlap = 0;
        for (auto k : neg)
            if (set.count(k)) ++overlap;
        if (overlap > 0)
            std::cerr << "[ERROR] " << a.filter
                      << " neg set overlaps with pos set! overlap="
                      << overlap << std::endl;
    }

    // ------------------------
    // Build and finalize filter
    // ------------------------
    FilterT f(a.n, a.fpr, a.seed);
    for (auto k : pos) f.insert(k);
    f.finalize();

    // ------------------------
    // Query generation (mixed positives/negatives)
    // ------------------------
    const size_t Q = a.n;
    const size_t neg_q = static_cast<size_t>(a.neg_ratio * Q);
    const size_t pos_q = Q - neg_q;

    std::shuffle(neg.begin(), neg.end(), gen);
    std::vector<uint64_t> queries;
    queries.reserve(Q);

    for (size_t i = 0; i < pos_q; ++i)
        queries.push_back(pos[(i * 9973) % pos.size()]);
    for (size_t i = 0; i < neg_q; ++i)
        queries.push_back(neg[(i * 7919) % neg.size()]);
    std::shuffle(queries.begin(), queries.end(), gen);

    // ------------------------
    // Verify negative lookup ratio sanity
    // ------------------------
    {
        size_t neg_count = 0, pos_count = 0;
        for (auto q : queries) {
            if (set.count(q))
                ++pos_count;
            else
                ++neg_count;
        }
        double actual_ratio = static_cast<double>(neg_count) / queries.size();
        std::cout << "[SANITY] " << a.filter
                  << " target_neg_ratio=" << a.neg_ratio
                  << " actual=" << actual_ratio
                  << "  pos=" << pos_count
                  << "  neg=" << neg_count
                  << std::endl;
    }

    // ------------------------
    // Optional: quick false-positive estimate
    // ------------------------
    {
        size_t neg_hits = 0;
        for (size_t i = 0; i < std::min<size_t>(neg.size(), 10000); ++i)
            if (f.contains(neg[i])) ++neg_hits;
        double fp_rate = static_cast<double>(neg_hits) /
                         std::min<size_t>(neg.size(), 10000);
        std::cout << "[SANITY] " << a.filter
                  << " FP rate (10k sample) = "
                  << fp_rate * 100.0 << "%" << std::endl;
    }

    // ------------------------
    // Latency + throughput measurement (ns precision)
    // ------------------------
    const size_t sample_stride = std::max<size_t>(Q / 20, 1); // sample ~5%
    std::vector<uint64_t> lat_ns;
    lat_ns.reserve((Q / sample_stride + 1) * a.reps);

    double total_t = 0.0;
    auto base_gen = gen; // deterministic repetition RNG

    for (int r = 0; r < a.reps; ++r) {
        gen = base_gen;  // identical ordering each repetition
        auto run_queries = queries;

        // ------------------------------------------------------------
        // Warm-up: use disjoint keys for realistic cache behavior
        // ------------------------------------------------------------
        if (a.warm) {
            for (auto q : run_queries)
                (void)f.contains(q ^ 0xdeadbeefULL);
        }
        auto start = high_resolution_clock::now();
        size_t idx = 0;
        for (auto q : run_queries) {
            if ((idx % sample_stride) == 0) {
                uint64_t t1 = now_ns();
                (void)f.contains(q);
                uint64_t t2 = now_ns();
                lat_ns.push_back(t2 - t1);
            } else {
                (void)f.contains(q);
            }
            ++idx;
        }
        auto end = high_resolution_clock::now();
        total_t += duration<double>(end - start).count();
    }

    // ------------------------
    // Aggregate metrics
    // ------------------------
    double qps = (Q * a.reps) / total_t;
    double p50_ns, p95_ns, p99_ns;
    quantiles(lat_ns, p50_ns, p95_ns, p99_ns);

    // Compute achieved FPR over full negative set
    double achieved = achieved_fpr(f, neg);

    // ------------------------
    // Write results (latency in ns + achieved FPR)
    // ------------------------
    csv << a.filter << "," << a.fpr << "," << achieved << "," << a.neg_ratio << ","
        << a.threads << "," << qps << ","
        << p50_ns << "," << p95_ns << "," << p99_ns << ","
        << a.seed << "\n";

    std::cout << "[LOOKUP] " << a.filter
            << " neg=" << a.neg_ratio
            << " qps=" << qps
            << " p50=" << p50_ns << "ns"
            << " p95=" << p95_ns << "ns"
            << " p99=" << p99_ns << "ns"
            << " fpr=" << achieved << "\n";
}


// =============================================================
// 3. Insert/Delete throughput (dynamic filters only)
// =============================================================
template <typename FilterT>
static void do_insert(const Args &a, std::ofstream &csv) {
    const size_t cap = a.n;

    // Key material
    std::mt19937_64 gen(a.seed ? a.seed : 9012);
    std::vector<uint64_t> keys(cap);
    for (auto &k : keys) k = rand64(gen);

    // Construct filter
    FilterT f(cap, a.fpr, a.seed);

    // ---- True capacity (items/slots) & target prefill ----
    size_t cap_items = 0;
    if constexpr (std::is_same_v<FilterT, CuckooFilter>) {
        cap_items = f.capacity_items();      // buckets * BUCKET_SIZE + STASH
    } else if constexpr (std::is_same_v<FilterT, QuotientFilter>) {
        cap_items = f.capacity_slots();      // total table slots
    } else {
        cap_items = cap;                     // fallback
    }
    const size_t target_prefill = static_cast<size_t>(a.load_factor * cap_items);

    // ---- Prefill to target occupancy ----
    size_t inserted = 0;
    for (size_t i = 0; i < target_prefill; ++i) {
        f.insert(keys[i % keys.size()]);
        ++inserted;
        if constexpr (std::is_same_v<FilterT, CuckooFilter>) {
            // Stop early if we start failing during prefill
            if (f.failed_inserts() > 0) break;
        }
    }

    if (inserted < target_prefill) {
        std::cerr << "[WARN] " << a.filter
                  << " prefill stopped early: " << inserted << "/"
                  << target_prefill << "\n";
    } else {
        std::cout << "[INFO] Prefill OK: " << inserted << " ("
                  << a.load_factor * 100.0 << "% of true capacity)\n";
    }

    // Prefill failure rate (for info only)
    double prefill_fail_rate = 0.0;
    if constexpr (std::is_same_v<FilterT, CuckooFilter>) {
        const size_t fails = f.failed_inserts();
        prefill_fail_rate = inserted ? static_cast<double>(fails) / inserted : 0.0;
        std::cout << "[VERIFY] Cuckoo prefill fail_rate="
                  << prefill_fail_rate * 100.0 << "%\n";
    }

    // ---- Reset metrics for clean timing ----
    f.reset_metrics();
    if constexpr (std::is_same_v<FilterT, QuotientFilter>) {
        f.set_metrics_enabled(true);
    }

    // ------------------------------------------------------------
    // Optional warm-up before timing
    // ------------------------------------------------------------
    if (a.warm) {
        // Perform a small batch of contains() to stabilize cache and branch predictors
        size_t warm_count = std::min<size_t>(cap_items / 100, 4096);
        for (size_t i = 0; i < warm_count; ++i)
            (void)f.contains(keys[i % keys.size()]);
    }


    // ---- Measurement window setup ----
    // Fixed-count inserts for stable measurements
    const size_t remaining = (cap_items > inserted) ? (cap_items - inserted) : 0;
    const size_t measure_n = std::min<size_t>(
        std::max<size_t>(cap_items / 100, 100'000),  // ~1% of capacity, at least 100k
        remaining > 0 ? remaining : 0
    );
    const size_t start_idx = inserted;

    // ---- Timed INSERT batch (fixed count) ----
    size_t cuckoo_fail_before = 0;
    if constexpr (std::is_same_v<FilterT, CuckooFilter>) {
        cuckoo_fail_before = f.failed_inserts();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < measure_n; ++i) {
        // modulo to stay in-bounds if keys < measure_n
        f.insert(keys[(start_idx + i) % keys.size()]);
    }
    const double ins_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    const double insert_ops_s = (measure_n > 0 && ins_sec > 0.0)
                                ? static_cast<double>(measure_n) / ins_sec
                                : 0.0;

    // Cuckoo failure delta during the measured inserts
    double timed_fail_rate = 0.0;
    if constexpr (std::is_same_v<FilterT, CuckooFilter>) {
        const size_t cuckoo_fail_after = f.failed_inserts();
        const size_t delta = (cuckoo_fail_after > cuckoo_fail_before)
                                 ? (cuckoo_fail_after - cuckoo_fail_before)
                                 : 0;
        timed_fail_rate = (measure_n > 0)
                              ? static_cast<double>(delta) / static_cast<double>(measure_n)
                              : 0.0;
    }

    // ---- Timed DELETE batch (half of prefill) ----
    const size_t del_n = std::min<size_t>(inserted / 2, target_prefill);
    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < del_n; ++i) {
        if constexpr (std::is_same_v<FilterT, CuckooFilter>) {
            f.remove(keys[i % keys.size()]);
        } else if constexpr (std::is_same_v<FilterT, QuotientFilter>) {
            f.remove(keys[i % keys.size()]);
        }
    }
    const double del_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    const double delete_ops_s = (del_n > 0 && del_sec > 0.0)
                                ? static_cast<double>(del_n) / del_sec
                                : 0.0;

    double avg_probe = 0.0, avg_cluster = 0.0, max_cluster = 0.0;
    size_t evictions = 0, stash = 0;

    if constexpr (std::is_same_v<FilterT, CuckooFilter>) {
        evictions = f.eviction_attempts();
        stash     = f.stash_hits();
    }
    else if constexpr (std::is_same_v<FilterT, QuotientFilter>) {
        avg_probe   = f.avg_probe_length();
        avg_cluster = f.avg_cluster_length();
        max_cluster = static_cast<double>(f.max_cluster_length());
    }

    // export cluster histogram bins (Quotient only)
    std::vector<size_t> hist;
    if constexpr (std::is_same_v<FilterT, QuotientFilter>) {
        hist = f.cluster_histogram();
    }

    // --- Histogram serialization (Quotient only) ---
    std::string hist_str;
    if constexpr (std::is_same_v<FilterT, QuotientFilter>) {
        std::ostringstream oss;
        const auto& hist = f.cluster_histogram();
        for (size_t i = 0; i < hist.size(); ++i) {
            oss << hist[i];
            if (i + 1 < hist.size()) oss << ",";
        }
        hist_str = oss.str();
    }

    csv << a.filter << ","
        << a.load_factor << ","
        << insert_ops_s << ","
        << delete_ops_s << ","
        << timed_fail_rate << ","
        << avg_probe << ","
        << avg_cluster << ","
        << max_cluster << ","
        << evictions << ","
        << stash;

    if constexpr (std::is_same_v<FilterT, QuotientFilter>) {
        csv << ",\"" << hist_str << "\"";
    }
    csv << "\n";

    std::cout << "[INSERT] " << a.filter
              << " load=" << a.load_factor
              << " ins="  << insert_ops_s
              << " del="  << delete_ops_s << "%\n";
}


// =============================================================
// 4. Thread scaling (read-mostly & balanced)  [MULTI-THREADED]
// =============================================================
template <typename FilterT>
static void do_threads(const Args &a, std::ofstream &csv) {
    // ---------------- Prep data ----------------
    std::mt19937_64 gen(a.seed ? a.seed : 3456);
    std::vector<uint64_t> pos(a.n), neg(a.n);
    for (auto &k : pos) k = rand64(gen);
    std::unordered_set<uint64_t> set(pos.begin(), pos.end());
    for (size_t i = 0; i < a.n; ++i) {
        uint64_t x; do { x = rand64(gen); } while (set.count(x));
        neg[i] = x;
    }

    FilterT f(a.n, a.fpr, a.seed);
    for (auto k : pos) f.insert(k);
    f.finalize();

    // Compose a single-thread query mix generator
    auto make_queries = [&](size_t Q, std::mt19937_64 &g) {
        std::vector<uint64_t> qv; qv.reserve(Q);
        if (a.workload == "read_mostly") {
            const size_t pos_q = static_cast<size_t>(Q * 0.90); // 90% lookups (read-heavy)
            for (size_t i = 0; i < pos_q; ++i) qv.push_back(pos[(i * 9973) % pos.size()]);
            for (size_t i = pos_q; i < Q;  ++i) qv.push_back(neg[(i * 7919) % neg.size()]);
        } else { // balanced 50/50
            const size_t pos_q = Q / 2;
            for (size_t i = 0; i < pos_q; ++i) qv.push_back(pos[(i * 9973) % pos.size()]);
            for (size_t i = pos_q; i < Q;  ++i) qv.push_back(neg[(i * 7919) % neg.size()]);
        }
        std::shuffle(qv.begin(), qv.end(), g);
        return qv;
    };

    // ---------------- Multi-thread run ----------------
    const int T = std::max(1, a.threads);
    const size_t Q_per_thread = a.n;                  // keep per-thread work constant
    const size_t TOTAL_Q = Q_per_thread * size_t(T);  // total ops increases with threads

    std::vector<std::thread> workers;
    workers.reserve(T);

    // per-thread latency samples (small to keep overhead low)
    std::vector<std::vector<uint64_t>> lat_samples(T);

    std::atomic<bool> go(false);

    // Launch workers
    for (int t = 0; t < T; ++t) {
        workers.emplace_back([&, t](){
            std::mt19937_64 tg(a.seed + 0x9e3779b97f4a7c15ULL * (t + 1));
            auto qv = make_queries(Q_per_thread, tg);

            // warm-up to stabilize caches
            if (a.warm) {
                for (size_t i = 0; i < std::min<size_t>(qv.size(), 4096); ++i)
                    (void)f.contains(qv[i] ^ 0xdeadbeefULL);
            }

            // wait for the global start
            while (!go.load(std::memory_order_acquire)) { /* spin */ }

            // sample roughly ~0.5% to keep overhead small
            const size_t stride = std::max<size_t>(Q_per_thread / 200, 1);
            auto &ls = lat_samples[t];
            ls.reserve(Q_per_thread / stride + 2);

            for (size_t i = 0; i < Q_per_thread; ++i) {
                if ((i % stride) == 0) {
#if defined(__linux__)
                    uint64_t t1 = now_ns();
                    (void)f.contains(qv[i]);
                    uint64_t t2 = now_ns();
                    ls.push_back(t2 - t1);
#else
                    auto t1 = std::chrono::high_resolution_clock::now();
                    (void)f.contains(qv[i]);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    ls.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
#endif
                } else {
                    (void)f.contains(qv[i]);
                }
            }
        });
    }

    // Global wall-clock measurement
    auto t0 = std::chrono::high_resolution_clock::now();
    go.store(true, std::memory_order_release);
    for (auto &th : workers) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double qps = (sec > 0.0) ? (double)TOTAL_Q / sec : 0.0;

    // Merge latency samples and compute quantiles (ns → us for CSV columns)
    std::vector<uint64_t> all_lat;
    size_t total_samples = 0; for (auto &v : lat_samples) total_samples += v.size();
    all_lat.reserve(total_samples);
    for (auto &v : lat_samples) { all_lat.insert(all_lat.end(), v.begin(), v.end()); }

    double p50_ns=0, p95_ns=0, p99_ns=0;
    quantiles(all_lat, p50_ns, p95_ns, p99_ns);

    // CSV: keep same schema as before (pXX_us columns)
    csv << a.filter << ","
        << a.workload << ","
        << a.threads << ","
        << qps << ","
        << (p50_ns / 1000.0) << ","
        << (p95_ns / 1000.0) << ","
        << (p99_ns / 1000.0) << ","
        << a.seed << "\n";

    std::cout << "[THREADS] " << a.filter
              << " wl=" << a.workload
              << " T=" << a.threads
              << " totalQ=" << TOTAL_Q
              << " qps=" << qps << std::endl;
}


// =============================================================
// MAIN
// =============================================================
int main(int argc, char **argv) {
    Args a = parse_args(argc, argv);
    std::ofstream csv(a.out, std::ios::app);
    if (!csv) { std::cerr << "[ERROR] cannot open " << a.out << "\n"; return 1; }

    std::cout << "=== bench mode=" << a.mode << " filter=" << a.filter
              << " n=" << a.n << " fpr=" << a.fpr << " out=" << a.out << " ===\n";

    if (a.mode == "space") {
        ensure_header(csv, "filter,n,target_fpr,achieved_fpr,bpe,fp_bits,table_bytes,meta_bytes,build_ms,seed");
        if (a.filter == "bloom") do_space<BloomFilter>(a, csv);
        else if (a.filter == "xor") do_space<XorFilter>(a, csv);
        else if (a.filter == "cuckoo") do_space<CuckooFilter>(a, csv);
        else if (a.filter == "quotient") do_space<QuotientFilter>(a, csv);
    } else if (a.mode == "lookup") {
        ensure_header(csv, "filter,target_fpr,achieved_fpr,neg_ratio,threads,qps,p50_ns,p95_ns,p99_ns,seed");
        if (a.filter == "bloom") do_lookup<BloomFilter>(a, csv);
        else if (a.filter == "xor") do_lookup<XorFilter>(a, csv);
        else if (a.filter == "cuckoo") do_lookup<CuckooFilter>(a, csv);
        else if (a.filter == "quotient") do_lookup<QuotientFilter>(a, csv);
    } else if (a.mode == "insert") {
        ensure_header(csv,
        "filter,load_factor,insert_ops_s,delete_ops_s,fail_rate,"
        "avg_probe_length,avg_cluster_length,max_cluster_length,"
        "evictions,stash_hits,cluster_hist");
        if (a.filter == "cuckoo") do_insert<CuckooFilter>(a, csv);
        else if (a.filter == "quotient") do_insert<QuotientFilter>(a, csv);
        else {
            std::cerr << "[WARN] insert mode only supports dynamic filters (cuckoo|quotient)\n";
        }
    } else if (a.mode == "threads") {
        ensure_header(csv, "filter,workload,threads,qps,p50_us,p95_us,p99_us,seed");
        if (a.filter == "bloom") do_threads<BloomFilter>(a, csv);
        else if (a.filter == "xor") do_threads<XorFilter>(a, csv);
        else if (a.filter == "cuckoo") do_threads<CuckooFilter>(a, csv);
        else if (a.filter == "quotient") do_threads<QuotientFilter>(a, csv);
    } else {
        std::cerr << "[ERROR] unknown mode " << a.mode << "\n";
        return 2;
    }

    return 0;
}
