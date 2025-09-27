// saxpy.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

// Stride SAXPY: each run executes exactly N*iters updates.
// Indexing uses (i * stride) % N to ensure constant work regardless of stride.
static inline void saxpy_stride(float* X, float* Y, float a, int N, int stride, int iters) {
    for (int r = 0; r < iters; ++r) {
        for (int i = 0; i < N; ++i) {
            int idx = (int)((1ULL * i * (unsigned)stride) % (unsigned)N);
            Y[idx] = a * X[idx] + Y[idx];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr <<
            "Usage: ./saxpy <N> <a> stride <iters> <stride>\n"
            "Example: ./saxpy 16777216 2.0 stride 4 9\n";
        return 1;
    }

    const int   N      = std::atoi(argv[1]);
    const float a      = std::atof(argv[2]);
    const std::string mode = argv[3];
    const int   iters  = std::atoi(argv[4]);
    const int   stride = std::atoi(argv[5]);

    if (mode != "stride") {
        std::cerr << "Error: only 'stride' mode is supported in this kernel.\n";
        return 1;
    }
    if (N <= 0 || iters <= 0 || stride <= 0) {
        std::cerr << "Error: N, iters, and stride must be positive.\n";
        return 1;
    }

    // Allocate & init
    std::vector<float> X(N), Y(N);
    for (int i = 0; i < N; ++i) {
        X[i] = float((i % 100) * 0.5f);
        Y[i] = float(((i % 50) - 25) * 0.25f);
    }

    // Warm-up (light run)
    saxpy_stride(X.data(), Y.data(), a, N, stride, 1);

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    saxpy_stride(X.data(), Y.data(), a, N, stride, iters);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "N=" << N
              << " a=" << a
              << " stride=" << stride
              << " iters=" << iters
              << " runtime_ms=" << ms
              << std::endl;

    return 0;
}
