#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <random>
#include <algorithm>
#include <cerrno>
#include <cstring>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sched.h>
#include <unistd.h>
#endif

// ========== Utility: pinning + timing ==========
void pin_process_to_core0() {
#if defined(_WIN32)
    HANDLE process = GetCurrentProcess();
    DWORD_PTR mask = 1; // first logical processor
    if (!SetProcessAffinityMask(process, mask)) {
        std::cerr << "Failed to set process affinity (Win32): " << GetLastError() << "\n";
    }
#else
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        std::cerr << "Failed to set CPU affinity: " << std::strerror(errno) << "\n";
    }
#endif
}

template <typename F>
double benchmark(F func) {
    auto t0 = std::chrono::high_resolution_clock::now();
    func();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    return dt.count();
}

// ========== Deterministic inputs ==========
template <typename T>
std::vector<T> generate_x(size_t N) {
    std::vector<T> v(N);
    for (size_t i = 0; i < N; ++i)
        v[i] = std::sin(double(i) * 0.001) * 0.5 + 0.5;
    return v;
}

template <typename T>
std::vector<T> generate_y(size_t N) {
    std::vector<T> v(N);
    for (size_t i = 0; i < N; ++i)
        v[i] = std::cos(double(i) * 0.001);
    return v;
}

// ========== Baseline kernels ==========
template <typename T>
void saxpy(size_t N, T a, const T* __restrict x, T* __restrict y) {
    for (size_t i = 0; i < N; ++i) y[i] = a * x[i] + y[i];
}

template <typename T>
T dot(size_t N, const T* __restrict x, const T* __restrict y) {
    T acc = 0;
    for (size_t i = 0; i < N; ++i) acc += x[i] * y[i];
    return acc;
}

template <typename T>
void elem_mul(size_t N, const T* __restrict x, const T* __restrict y, T* __restrict z) {
    for (size_t i = 0; i < N; ++i) z[i] = x[i] * y[i];
}

template <typename T>
void stencil(size_t N, const T* __restrict x, T* __restrict y, T a, T b, T c) {
    if (N == 0) return;
    y[0] = x[0];
    for (size_t i = 1; i + 1 < N; ++i) {
        y[i] = a * x[i - 1] + b * x[i] + c * x[i + 1];
    }
    if (N > 1) y[N - 1] = x[N - 1];
}

// ========== Strided kernels ==========
template <typename T>
T strided_dot(const T* __restrict x, const T* __restrict y, size_t N, size_t s) {
    T acc = 0;
    for (size_t i = 0; i < N; i += s) acc += x[i] * y[i];
    return acc;
}

template <typename T>
void strided_saxpy(T* __restrict y, const T* __restrict x, size_t N, size_t s, T a) {
    for (size_t i = 0; i < N; i += s) y[i] = a * x[i] + y[i];
}

template <typename T>
void strided_elem_mul(T* __restrict z, const T* __restrict x, const T* __restrict y, size_t N, size_t s) {
    for (size_t i = 0; i < N; i += s) z[i] = x[i] * y[i];
}

template <typename T>
void strided_stencil(T* __restrict y, const T* __restrict x, size_t N, size_t s, T a, T b, T c) {
    if (N == 0) return;
    // Keep boundary same as baseline for consistency
    if (N >= 1) y[0] = x[0];
    if (N >= 2) y[N - 1] = x[N - 1];
    for (size_t i = 1; i + 1 < N; i += s) {
        y[i] = a * x[i - 1] + b * x[i] + c * x[i + 1];
    }
}

// ========== Gather kernels (idx-based) ==========
template <typename T>
T gather_dot(const T* __restrict x, const T* __restrict y, const std::vector<size_t>& idx) {
    T acc = 0;
    for (size_t j : idx) acc += x[j] * y[j];
    return acc;
}

template <typename T>
void gather_saxpy(T* __restrict y, const T* __restrict x, const std::vector<size_t>& idx, T a) {
    for (size_t j : idx) y[j] = a * x[j] + y[j];
}

template <typename T>
void gather_elem_mul(T* __restrict z, const T* __restrict x, const T* __restrict y, const std::vector<size_t>& idx) {
    for (size_t j : idx) z[j] = x[j] * y[j];
}

template <typename T>
void gather_stencil(T* __restrict y, const T* __restrict x, size_t N, const std::vector<size_t>& idx, T a, T b, T c) {
    // Only compute where neighbors exist; skip boundary indices.
    for (size_t j : idx) {
        if (j > 0 && j + 1 < N) {
            y[j] = a * x[j - 1] + b * x[j] + c * x[j + 1];
        }
    }
}

// ========== Main driver ==========
int main(int argc, char** argv) {
    pin_process_to_core0();

    size_t N = (argc > 1) ? std::stoul(argv[1]) : static_cast<size_t>(1e6);
    size_t offset_elems = (argc > 2) ? std::stoul(argv[2]) : 0;
    std::string mode = (argc > 3) ? argv[3] : "baseline";

    // Parameters for special modes
    size_t stride = 1;
    int gather_mode = 0;      // 1 = blocked (contiguous sub-blocks), 2 = random
    double gather_frac = 1.0; // fraction of N used by gather

    if (mode == "stride") {
        if (argc > 4) stride = std::stoul(argv[4]);
    } else if (mode == "gather") {
        if (argc > 4) gather_mode = std::stoi(argv[4]);
        if (argc > 5) gather_frac = std::stod(argv[5]);
        if (gather_frac <= 0.0) gather_frac = 1.0;
        if (gather_frac > 1.0) gather_frac = 1.0;
    }

    if (mode == "baseline") {
        // -------- Float32 --------
        {
            auto x_full = generate_x<float>(N + offset_elems + 1);
            auto y_full = generate_y<float>(N + offset_elems + 1);
            auto z_full = std::vector<float>(N + offset_elems + 1);

            float* x = x_full.data() + offset_elems;
            float* y = y_full.data() + offset_elems;
            float* z = z_full.data() + offset_elems;
            float a = 2.5f;

            double t = benchmark([&]{ saxpy(N, a, x, y); });
            std::cout << "SAXPY time = " << t << " s | checksum(y) = " << (y[0] + y[N/2] + y[N-1]) << "\n";

            float res;
            t = benchmark([&]{ volatile float rr = dot(N, x, y); });
            res = dot(N, x, y);
            std::cout << "Dot time = " << t << " s | result = " << res << "\n";

            t = benchmark([&]{ elem_mul(N, x, y, z); });
            std::cout << "ElemMul time = " << t << " s | checksum(z) = " << (z[0] + z[N/2] + z[N-1]) << "\n";

            t = benchmark([&]{ stencil(N, x, z, 0.25f, 0.5f, 0.25f); });
            std::cout << "Stencil time = " << t << " s | checksum(z) = " << (z[0] + z[N/2] + z[N-1]) << "\n";
        }

        // -------- Float64 --------
        {
            auto x_full = generate_x<double>(N + offset_elems + 1);
            auto y_full = generate_y<double>(N + offset_elems + 1);
            auto z_full = std::vector<double>(N + offset_elems + 1);

            double* x = x_full.data() + offset_elems;
            double* y = y_full.data() + offset_elems;
            double* z = z_full.data() + offset_elems;
            double a = 2.5;

            double t = benchmark([&]{ saxpy(N, a, x, y); });
            std::cout << "SAXPY time = " << t << " s | checksum(y) = " << (y[0] + y[N/2] + y[N-1]) << "\n";

            double res;
            t = benchmark([&]{ volatile double rr = dot(N, x, y); });
            res = dot(N, x, y);
            std::cout << "Dot time = " << t << " s | result = " << res << "\n";

            t = benchmark([&]{ elem_mul(N, x, y, z); });
            std::cout << "ElemMul time = " << t << " s | checksum(z) = " << (z[0] + z[N/2] + z[N-1]) << "\n";

            t = benchmark([&]{ stencil(N, x, z, 0.25, 0.5, 0.25); });
            std::cout << "Stencil time = " << t << " s | checksum(z) = " << (z[0] + z[N/2] + z[N-1]) << "\n";
        }
    }
    else if (mode == "stride") {
        // Use offset’d views to keep compatibility with alignment/tail experiments
        auto x_full = generate_x<float>(N + offset_elems + 1);
        auto y_full = generate_y<float>(N + offset_elems + 1);
        auto z_full = std::vector<float>(N + offset_elems + 1);

        float* x = x_full.data() + offset_elems;
        float* y = y_full.data() + offset_elems;
        float* z = z_full.data() + offset_elems;

        float a = 2.5f;

        // Dot
        double t = benchmark([&]{ volatile float rr = strided_dot(x, y, N, stride); });
        std::cout << "Dot time = " << t << " s | = " << (x[0] * y[0]) << "\n";

        // SAXPY
        t = benchmark([&]{ strided_saxpy(y, x, N, stride, a); });
        std::cout << "SAXPY time = " << t << " s | = " << (y[0] + y[N/2] + y[N-1]) << "\n";

        // ElemMul
        t = benchmark([&]{ strided_elem_mul(z, x, y, N, stride); });
        std::cout << "ElemMul time = " << t << " s | = " << (z[0] + z[N/2] + z[N-1]) << "\n";

        // Stencil
        t = benchmark([&]{ strided_stencil(z, x, N, stride, 0.25f, 0.5f, 0.25f); });
        std::cout << "Stencil time = " << t << " s | = " << (z[0] + z[N/2] + z[N-1]) << "\n";
    }
    else if (mode == "gather") {
        // Offset’d views
        auto x_full = generate_x<float>(N + offset_elems + 1);
        auto y_full = generate_y<float>(N + offset_elems + 1);
        auto z_full = std::vector<float>(N + offset_elems + 1);

        float* x = x_full.data() + offset_elems;
        float* y = y_full.data() + offset_elems;
        float* z = z_full.data() + offset_elems;

        float a = 2.5f;

        // Build indices
        size_t M = std::max<size_t>(1, size_t(N * gather_frac));
        std::vector<size_t> idx; idx.reserve(M);

        if (gather_mode == 1) {
            // "blocked gather": take small contiguous runs separated by gaps
            size_t j = 0;
            for (size_t base = 0; j < M && base < N; base += 64) {
                for (int k = 0; k < 16 && j < M && base + size_t(k) < N; ++k) {
                    idx.push_back(base + size_t(k));
                    ++j;
                }
            }
        } else {
            // random-ish indices
            idx.resize(M);
            for (size_t i = 0; i < M; ++i) idx[i] = (i * 9973) % N;
            std::mt19937_64 rng(42);
            std::shuffle(idx.begin(), idx.end(), rng);
        }

        // Dot
        double t = benchmark([&]{ volatile float rr = gather_dot(x, y, idx); });
        std::cout << "Dot time = " << t << " s | = " << (x[idx.empty()?0:idx[0]] * y[idx.empty()?0:idx[0]]) << "\n";

        // SAXPY
        t = benchmark([&]{ gather_saxpy(y, x, idx, a); });
        std::cout << "SAXPY time = " << t << " s | = " << (y[0] + y[N/2] + y[N-1]) << "\n";

        // ElemMul
        t = benchmark([&]{ gather_elem_mul(z, x, y, idx); });
        std::cout << "ElemMul time = " << t << " s | = " << (z[0] + z[N/2] + z[N-1]) << "\n";

        // Stencil (compute only where neighbors exist)
        t = benchmark([&]{ gather_stencil(z, x, N, idx, 0.25f, 0.5f, 0.25f); });
        std::cout << "Stencil time = " << t << " s | = " << (z[0] + z[N/2] + z[N-1]) << "\n";
    }

    return 0;
}
