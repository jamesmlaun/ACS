#!/usr/bin/env python3
import subprocess, datetime, os, csv, re, statistics, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SRC_FILE = "kernels.cpp"
SCALAR_EXE = "kernels_scalar"
SIMD_EXE   = "kernels_simd"

# Warmup flag (set via project_manager.py -> command line)
WARMUP_CACHE = ("--warmup" in sys.argv)

LOG_DIR, CSV_DIR, PLOT_DIR = "logs", "results", "plots"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cache_state = "warm" if WARMUP_CACHE else "cold"
logfile = os.path.join(LOG_DIR, f"baseline_{ts}_{cache_state}.log")
csvfile = os.path.join(CSV_DIR, f"baseline_{ts}_{cache_state}.csv")
plotdir = os.path.join(PLOT_DIR, f"baseline_{ts}_{cache_state}")
os.makedirs(plotdir, exist_ok=True)

# === CPU cache sizes (per-core) ===
L1_BYTES = 48 * 1024
L2_BYTES = 1280 * 1024
L3_BYTES = 24 * 1024 * 1024
ELEM_SIZE = 4  # float32

ARRAYS_PER_KERNEL = {"SAXPY":2,"Dot":2,"ElemMul":3,"Stencil":2}
FLOPS_PER_ELEM = {"SAXPY":2,"Dot":2,"ElemMul":1,"Stencil":3}
RUNS_PER_N = 3

# Validation tolerances
RTOL = 1e-5
ATOL = 1e-7

def run_cmd(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout

def parse_output(out):
    results = {}
    values = {}
    for line in out.splitlines():
        m = re.match(r"(\w+) time = ([0-9.eE+-]+) s.*= ([0-9.eE+-]+)", line)
        if m:
            kernel = m.group(1)
            results[kernel] = float(m.group(2))
            values[kernel]  = float(m.group(3))
    return results, values

def validate(s_val, v_val):
    abs_err = abs(v_val - s_val)
    rel_err = abs_err / max(abs(s_val), 1e-30)
    ok = abs_err <= ATOL or rel_err <= RTOL
    return ok, abs_err, rel_err

def n_midpoints(arrays, elem_size=ELEM_SIZE):
    N_L1_mid   = (L1_BYTES//2) // (arrays*elem_size)
    N_L2_mid   = ((L1_BYTES+L2_BYTES)//2) // (arrays*elem_size)
    N_L3_mid   = ((L2_BYTES+L3_BYTES)//2) // (arrays*elem_size)
    N_DRAM_mid = ((L3_BYTES*2 + L3_BYTES*8)//2) // (arrays*elem_size)
    return {"L1_mid":N_L1_mid, "L2_mid":N_L2_mid, "L3_mid":N_L3_mid, "DRAM_mid":N_DRAM_mid}

def n_for_cache(cache_bytes, arrays, elem_size=ELEM_SIZE):
    return cache_bytes // (arrays*elem_size)

def cache_thresholds(kernel):
    arrays = ARRAYS_PER_KERNEL[kernel]
    return {
        "L1": n_for_cache(L1_BYTES, arrays),
        "L2": n_for_cache(L2_BYTES, arrays),
        "L3": n_for_cache(L3_BYTES, arrays),
    }

with open(logfile, "w") as f, open(csvfile, "w", newline="") as cf:
    def log(msg, quiet=False):
        if not quiet:
            print(msg)
        f.write(msg + "\n")

    writer = csv.writer(cf)
    writer.writerow([
        "kernel","cache_region","N",
        "runtime_scalar_mean","runtime_scalar_std",
        "runtime_simd_mean","runtime_simd_std",
        "speedup",
        "gflops_scalar","gflops_simd",
        "runtime_scalar_median","runtime_scalar_var",
        "runtime_simd_median","runtime_simd_var",
        "speedup_median","speedup_var",
        "gflops_scalar_median","gflops_scalar_var",
        "gflops_simd_median","gflops_simd_var",
        "validated","rtol","atol","abs_err","rel_err"
    ])

    # === System info ===
    log("=== Baseline & Correctness Experiment ===")
    log(f"Timestamp: {ts}")
    log(f"Warmup enabled: {WARMUP_CACHE}")
    log("\n--- System Info ---")
    log(run_cmd(["lscpu"]))
    log("\n--- Compiler Info ---")
    log(run_cmd(["gcc","--version"]))

    # === Compilation ===
    log("\n[Compiling scalar baseline]")
    log(run_cmd(["g++","-O1","-fno-tree-vectorize","-std=c++17","-o",SCALAR_EXE,SRC_FILE]))

    log("\n[Compiling SIMD version]")
    log(run_cmd(["g++","-O3","-march=native","-ffast-math","-fopt-info-vec","-std=c++17","-o",SIMD_EXE,SRC_FILE]))

    # === Experiment runs ===
    for kernel in FLOPS_PER_ELEM:
        Ns = n_midpoints(ARRAYS_PER_KERNEL[kernel])
        for region, N in Ns.items():
            log(f"\n--- {kernel}: running {region} (N={N}) ---")
            s_times, v_times = [], []
            s_vals, v_vals = [], []

            num_runs = RUNS_PER_N
            warmup_extra = 1 if WARMUP_CACHE else 0

            for run in range(num_runs + warmup_extra):
                s_out = run_cmd([f"./{SCALAR_EXE}", str(N)])
                v_out = run_cmd([f"./{SIMD_EXE}", str(N)])

                # skip the warmup run
                if WARMUP_CACHE and run == 0:
                    continue

                s_res, s_valdict = parse_output(s_out)
                v_res, v_valdict = parse_output(v_out)

                if kernel in s_res: s_times.append(s_res[kernel])
                if kernel in v_res: v_times.append(v_res[kernel])
                if kernel in s_valdict: s_vals.append(s_valdict[kernel])
                if kernel in v_valdict: v_vals.append(v_valdict[kernel])

            if s_times and v_times:
                t_s_mean = statistics.mean(s_times)
                t_s_std  = statistics.stdev(s_times) if len(s_times)>1 else 0.0
                t_v_mean = statistics.mean(v_times)
                t_v_std  = statistics.stdev(v_times) if len(v_times)>1 else 0.0
                sp = t_s_mean/t_v_mean if t_v_mean>0 else float("nan")

                t_s_median = statistics.median(s_times)
                t_s_var = statistics.variance(s_times) if len(s_times)>1 else 0.0
                t_v_median = statistics.median(v_times)
                t_v_var = statistics.variance(v_times) if len(v_times)>1 else 0.0

                # GFLOP/s
                flops = FLOPS_PER_ELEM[kernel] * N
                gflops_scalar = flops / (t_s_mean * 1e9) if t_s_mean > 0 else float("nan")
                gflops_simd   = flops / (t_v_mean * 1e9) if t_v_mean > 0 else float("nan")

                gflops_scalar_runs = [flops / (t * 1e9) for t in s_times if t > 0]
                gflops_simd_runs = [flops / (t * 1e9) for t in v_times if t > 0]
                gflops_scalar_median = statistics.median(gflops_scalar_runs) if gflops_scalar_runs else float("nan")
                gflops_scalar_var = statistics.variance(gflops_scalar_runs) if len(gflops_scalar_runs)>1 else 0.0
                gflops_simd_median = statistics.median(gflops_simd_runs) if gflops_simd_runs else float("nan")
                gflops_simd_var = statistics.variance(gflops_simd_runs) if len(gflops_simd_runs)>1 else 0.0

                # Validation
                s_val_mean = statistics.mean(s_vals) if s_vals else float("nan")
                v_val_mean = statistics.mean(v_vals) if v_vals else float("nan")
                validated, abs_err, rel_err = validate(s_val_mean, v_val_mean)

                speedup_runs = [s/v for s, v in zip(s_times, v_times) if v > 0]
                sp_median = statistics.median(speedup_runs) if speedup_runs else float("nan")
                sp_var = statistics.variance(speedup_runs) if len(speedup_runs)>1 else 0.0

                writer.writerow([
                    kernel,region,N,
                    f"{t_s_mean:.6e}",f"{t_s_std:.6e}",
                    f"{t_v_mean:.6e}",f"{t_v_std:.6e}",
                    f"{sp:.3f}",
                    f"{gflops_scalar:.3f}",f"{gflops_simd:.3f}",
                    f"{t_s_median:.6e}",f"{t_s_var:.6e}",
                    f"{t_v_median:.6e}",f"{t_v_var:.6e}",
                    f"{sp_median:.3f}",f"{sp_var:.6e}",
                    f"{gflops_scalar_median:.3f}",f"{gflops_scalar_var:.6e}",
                    f"{gflops_simd_median:.3f}",f"{gflops_simd_var:.6e}",
                    int(validated), RTOL, ATOL,
                    f"{abs_err:.3e}", f"{rel_err:.3e}"
                ])

                log(f"Runtime scalar={t_s_mean:.3e}s, SIMD={t_v_mean:.3e}s, speedup={sp:.2f}")
                log(f"GFLOP/s scalar={gflops_scalar:.2f}, SIMD={gflops_simd:.2f}")
                log(f"Validation {kernel} {region} N={N}: "
                    f"scalar_val={s_val_mean:.6e}, simd_val={v_val_mean:.6e}, "
                    f"validated={validated}, rel_err={rel_err:.2e}")

# === Plotting ===
df = pd.read_csv(csvfile)

kernel_plot_data = {}

for kernel in df["kernel"].unique():
    sub = df[df.kernel==kernel].copy()
    thresholds = cache_thresholds(kernel)

    # Use median and variance for plotting
    sub["runtime_scalar_mean"] = sub["runtime_scalar_median"].astype(float)
    sub["runtime_scalar_std"] = sub["runtime_scalar_var"].astype(float)
    sub["runtime_simd_mean"] = sub["runtime_simd_median"].astype(float)
    sub["runtime_simd_std"] = sub["runtime_simd_var"].astype(float)
    sub["gflops_scalar"] = sub["gflops_scalar_median"].astype(float)
    sub["gflops_scalar_err"] = sub["gflops_scalar_var"].astype(float)
    sub["gflops_simd"] = sub["gflops_simd_median"].astype(float)
    sub["gflops_simd_err"] = sub["gflops_simd_var"].astype(float)
    sub["speedup"] = sub["speedup_median"].astype(float)
    sub["speedup_err"] = sub["speedup_var"].astype(float)

    kernel_plot_data[kernel] = sub

    # Runtime plot
    plt.figure()
    plt.errorbar(sub["N"].to_numpy(),
                 sub["runtime_scalar_mean"].astype(float).to_numpy(),
                 yerr=sub["runtime_scalar_std"].astype(float).to_numpy(),
                 marker="o", label="Scalar")
    plt.errorbar(sub["N"].to_numpy(),
                 sub["runtime_simd_mean"].astype(float).to_numpy(),
                 yerr=sub["runtime_simd_std"].astype(float).to_numpy(),
                 marker="o", label="SIMD")
    for lvl, Nth in thresholds.items():
        plt.axvline(Nth, color="gray", linestyle="--", alpha=0.7)
        plt.text(Nth, plt.ylim()[1]*0.9, lvl, rotation=90,
                 va="bottom", ha="right", fontsize=8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Runtime (s)")
    plt.title(f"{kernel}: Runtime (mid-cache points)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"{kernel}_runtime.png"))
    plt.close()

    # Speedup plot
    plt.figure()
    plt.errorbar(sub["N"].to_numpy(),
                 sub["speedup"].astype(float).to_numpy(),
                 yerr=sub["speedup_err"].astype(float).to_numpy(),
                 marker="s", label="Speedup")
    for lvl, Nth in thresholds.items():
        plt.axvline(Nth, color="gray", linestyle="--", alpha=0.7)
        plt.text(Nth, plt.ylim()[1]*0.9, lvl, rotation=90,
                 va="bottom", ha="right", fontsize=8)
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("Speedup (Scalar / SIMD)")
    plt.title(f"{kernel}: Speedup (mid-cache points)")
    plt.axhline(1.0, color="gray", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"{kernel}_speedup.png"))
    plt.close()

    # GFLOP/s plot
    plt.figure()
    plt.errorbar(sub["N"].to_numpy(),
                 sub["gflops_scalar"].astype(float).to_numpy(),
                 yerr=sub["gflops_scalar_err"].to_numpy(),
                 marker="o", label="Scalar")
    plt.errorbar(sub["N"].to_numpy(),
                 sub["gflops_simd"].astype(float).to_numpy(),
                 yerr=sub["gflops_simd_err"].to_numpy(),
                 marker="o", label="SIMD")
    for lvl, Nth in thresholds.items():
        plt.axvline(Nth, color="gray", linestyle="--", alpha=0.7)
        plt.text(Nth, plt.ylim()[1]*0.9, lvl, rotation=90,
                 va="bottom", ha="right", fontsize=8)
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("GFLOP/s")
    plt.title(f"{kernel}: Throughput (GFLOP/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"{kernel}_gflops.png"))
    plt.close()


if kernel_plot_data:
    cmap = plt.cm.get_cmap("tab10", len(kernel_plot_data))
    plt.figure()
    for idx, (kernel, sub) in enumerate(kernel_plot_data.items()):
        color = cmap(idx)
        ns = sub["N"].astype(float).to_numpy()
        scalar_runtime = sub["runtime_scalar_mean"].astype(float).to_numpy()
        simd_runtime = sub["runtime_simd_mean"].astype(float).to_numpy()
        scalar_err = sub["runtime_scalar_std"].astype(float).to_numpy()
        simd_err = sub["runtime_simd_std"].astype(float).to_numpy()

        plt.errorbar(ns, scalar_runtime, yerr=scalar_err, marker="o",
                     linestyle="-", color=color, label=f"{kernel} Scalar")
        plt.errorbar(ns, simd_runtime, yerr=simd_err, marker="o",
                     linestyle="--", color=color, label=f"{kernel} SIMD")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Comparison Across Kernels")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "all_kernels_runtime.png"))
    plt.close()

    plt.figure()
    for idx, (kernel, sub) in enumerate(kernel_plot_data.items()):
        color = cmap(idx)
        ns = sub["N"].astype(float).to_numpy()
        scalar_gflops = sub["gflops_scalar"].astype(float).to_numpy()
        simd_gflops = sub["gflops_simd"].astype(float).to_numpy()
        scalar_err = sub["gflops_scalar_err"].astype(float).to_numpy()
        simd_err = sub["gflops_simd_err"].astype(float).to_numpy()

        plt.errorbar(ns, scalar_gflops, yerr=scalar_err, marker="o",
                     linestyle="-", color=color, label=f"{kernel} Scalar")
        plt.errorbar(ns, simd_gflops, yerr=simd_err, marker="o",
                     linestyle="--", color=color, label=f"{kernel} SIMD")

    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("GFLOP/s")
    plt.title("GFLOP/s Comparison Across Kernels")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "all_kernels_gflops.png"))
    plt.close()

    plt.figure()
    for idx, (kernel, sub) in enumerate(kernel_plot_data.items()):
        color = cmap(idx)
        ns = sub["N"].astype(float).to_numpy()
        speedup = sub["speedup"].astype(float).to_numpy()
        speedup_err = sub["speedup_err"].astype(float).to_numpy()

        plt.errorbar(ns, speedup, yerr=speedup_err, marker="s",
                     linestyle="-", color=color, label=kernel)

    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("Speedup (Scalar / SIMD)")
    plt.title("Speedup Across Kernels")
    plt.axhline(1.0, color="gray", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, "all_kernels_speedup.png"))
    plt.close()


print(f"Baseline log: {logfile}")
print(f"Baseline results: {csvfile}")
print(f"Baseline plots in {plotdir}/")
