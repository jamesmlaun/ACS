// saxpy_tlb.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>
#include <sys/mman.h>

// ---- helpers ----
static void log_stderr(const char* msg) {
    std::fprintf(stderr, "%s", msg);
}

static void print_file_to_stderr(const char* path, const char* prefix=nullptr, int max_lines=0) {
    FILE* f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "[WARN] cannot open %s\n", path);
        return;
    }
    char buf[4096];
    int lines = 0;
    while (std::fgets(buf, sizeof(buf), f)) {
        if (max_lines && lines >= max_lines) break;
        if (prefix) std::fputs(prefix, stderr);
        std::fputs(buf, stderr);
        ++lines;
    }
    std::fclose(f);
}

static size_t sum_anon_huge_kb_smaps() {
    FILE* f = std::fopen("/proc/self/smaps", "r");
    if (!f) return 0;
    char buf[4096];
    size_t total_kb = 0;
    while (std::fgets(buf, sizeof(buf), f)) {
        // Lines look like: "AnonHugePages:     2048 kB"
        if (std::strncmp(buf, "AnonHugePages:", 14) == 0) {
            // extract integer (kB)
            char* p = buf + 14;
            while (*p && (*p < '0' || *p > '9')) ++p;
            long val = std::strtol(p, nullptr, 10);
            if (val > 0) total_kb += (size_t)val;
        }
    }
    std::fclose(f);
    return total_kb; // in kB
}

// ---- kernel ----
static inline void saxpy_tlb(float* X, float* Y, float a,
                             int pages, int elems_per_page,
                             int lines_per_page, int elems_per_line,
                             int passes, const std::string& pattern) {
    std::vector<int> page_order(pages);
    for (int p = 0; p < pages; ++p) page_order[p] = p;

    std::vector<int> line_offsets;
    for (int l = 0; l < lines_per_page; ++l) {
        int off = l * elems_per_line;
        if (off < elems_per_page) line_offsets.push_back(off);
    }
    if (line_offsets.empty()) line_offsets.push_back(0);

    std::mt19937 rng(0xC0FFEE);

    for (int pass = 0; pass < passes; ++pass) {
        if (pattern == "random") {
            std::shuffle(page_order.begin(), page_order.end(), rng);
        }
        for (int p : page_order) {
            int base = p * elems_per_page;
            for (int off : line_offsets) {
                int idx = base + off;
                Y[idx] = a * X[idx] + Y[idx];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr <<
            "Usage: ./saxpy_tlb <pages> <page_bytes> <lines_per_page> <line_bytes> "
            "<passes> <a> <pattern> <huge_pages>\n"
            "Example: ./saxpy_tlb 1024 4096 1 64 100 2.0 random off\n";
        return 1;
    }

    const int pages          = std::atoi(argv[1]);
    const int page_bytes     = std::atoi(argv[2]);
    const int lines_per_page = std::atoi(argv[3]);
    const int line_bytes     = std::atoi(argv[4]);
    const int passes         = std::atoi(argv[5]);
    const float a            = std::atof(argv[6]);
    const std::string pattern    = argv[7];
    const std::string huge_pages = argv[8];

    if (pages <= 0 || page_bytes <= 0 || lines_per_page <= 0 ||
        line_bytes <= 0 || passes <= 0) {
        std::cerr << "Error: all numeric arguments must be positive.\n";
        return 1;
    }

    const int elems_per_page = page_bytes / (int)sizeof(float);
    const int total_elems    = pages * elems_per_page;
    const int elems_per_line = line_bytes / (int)sizeof(float);

    std::vector<float> X(total_elems), Y(total_elems);
    for (int i = 0; i < total_elems; ++i) {
        X[i] = float((i % 100) * 0.5f);
        Y[i] = float(((i % 50) - 25) * 0.25f);
    }

    // --- THP diagnostics BEFORE hint ---
    std::fprintf(stderr, "[THP] ----- BEFORE madvise -----\n");
    std::fprintf(stderr, "[THP] enabled: ");
    print_file_to_stderr("/sys/kernel/mm/transparent_hugepage/enabled", nullptr, 1);
    std::fprintf(stderr, "[THP] defrag : ");
    print_file_to_stderr("/sys/kernel/mm/transparent_hugepage/defrag",  nullptr, 1);

    // show system-wide AnonHugePages
    std::fprintf(stderr, "[THP] /proc/meminfo AnonHugePages (first 5 lines shown around it):\n");
    // crude but effective: just dump the whole file; parser reads stdout only
    print_file_to_stderr("/proc/meminfo", "  ", 0);

    size_t smaps_anon_kb_before = sum_anon_huge_kb_smaps();
    std::fprintf(stderr, "[THP] /proc/self/smaps AnonHugePages total: %zu kB (before)\n",
                 smaps_anon_kb_before);

    // --- Apply THP hint per user arg ---
    if (huge_pages == "on") {
        madvise(X.data(), (size_t)total_elems * sizeof(float), MADV_HUGEPAGE);
        madvise(Y.data(), (size_t)total_elems * sizeof(float), MADV_HUGEPAGE);
        std::fprintf(stderr, "[THP] madvise: MADV_HUGEPAGE\n");
    } else {
        madvise(X.data(), (size_t)total_elems * sizeof(float), MADV_NOHUGEPAGE);
        madvise(Y.data(), (size_t)total_elems * sizeof(float), MADV_NOHUGEPAGE);
        std::fprintf(stderr, "[THP] madvise: MADV_NOHUGEPAGE\n");
    }

    // Touch once per page to give THP promotion a chance
    for (int p = 0; p < pages; ++p) {
        X[p * elems_per_page] += 0.0f;
        Y[p * elems_per_page] += 0.0f;
    }

    // --- THP diagnostics AFTER hint ---
    size_t smaps_anon_kb_after = sum_anon_huge_kb_smaps();
    std::fprintf(stderr, "[THP] /proc/self/smaps AnonHugePages total: %zu kB (after)\n",
                 smaps_anon_kb_after);

    // --- Warm-up ---
    saxpy_tlb(X.data(), Y.data(), a, pages, elems_per_page,
              lines_per_page, elems_per_line, 1, pattern);

    // --- Timed run ---
    auto t0 = std::chrono::high_resolution_clock::now();
    saxpy_tlb(X.data(), Y.data(), a, pages, elems_per_page,
              lines_per_page, elems_per_line, passes, pattern);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double updates = double(passes) * double(pages) * double(std::max(1, (int)std::ceil((double)lines_per_page)));
    const double gflops  = (updates * 2.0) / (ms * 1e6);

    // stdout (parsed by your Python)
    std::cout << "runtime_ms=" << ms << "\n";
    std::cout << "gflops=" << gflops << "\n";

    return 0;
}
