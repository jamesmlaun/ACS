#!/usr/bin/env python3
import subprocess, datetime, os, csv, re, statistics, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SRC_FILE = "kernels.cpp"
SCALAR_EXE = "kernels_scalar"
SIMD_EXE   = "kernels_simd"

# Warmup flag (set via project_manager.py)
WARMUP_CACHE = ("--warmup" in sys.argv)

LOG_DIR, CSV_DIR, PLOT_DIR = "logs", "results", "plots"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cache_state = "warm" if WARMUP_CACHE else "cold"
logfile = os.path.join(LOG_DIR, f"locality_{ts}_{cache_state}.log")
csvfile = os.path.join(CSV_DIR, f"locality_{ts}_{cache_state}.csv")
plotdir = os.path.join(PLOT_DIR, f"locality_{ts}_{cache_state}")
os.makedirs(plotdir, exist_ok=True)

RUNS_PER_N = 3
RTOL, ATOL = 1e-5, 1e-7

# Cache + kernel info
L1_BYTES = 48 * 1024
L2_BYTES = 1280 * 1024
L3_BYTES = 24 * 1024 * 1024
ELEM_SIZE = 4  # float32
ARRAYS_PER_KERNEL = {"SAXPY":2,"Dot":2,"ElemMul":3,"Stencil":2}
FLOPS_PER_ELEM = {"SAXPY":2,"Dot":2,"ElemMul":1,"Stencil":3}

# --- CPU frequency detection for CPE ---
def detect_cpu_freq():
    # Try lscpu
    try:
        lscpu_out = subprocess.run(["lscpu"], check=True, stdout=subprocess.PIPE, text=True).stdout
        m = re.search(r"CPU max MHz:\s*([\d.]+)", lscpu_out)
        if not m:
            m = re.search(r"CPU MHz:\s*([\d.]+)", lscpu_out)
        if m:
            return float(m.group(1)) * 1e6
    except Exception:
        pass

    # Try /proc/cpuinfo (works in WSL)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "cpu MHz" in line:
                    mhz = float(line.split(":")[1].strip())
                    return mhz * 1e6
    except Exception:
        pass

    # Fallback
    return 3.0e9  # default 3 GHz

CPU_FREQ_HZ = detect_cpu_freq()

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

def n_for_cache(cache_bytes, arrays, elem_size=ELEM_SIZE):
    return cache_bytes // (arrays * elem_size)

def cache_thresholds(kernel):
    arrays = ARRAYS_PER_KERNEL[kernel]
    return {
        "L1": n_for_cache(L1_BYTES, arrays),
        "L2": n_for_cache(L2_BYTES, arrays),
        "L3": n_for_cache(L3_BYTES, arrays),
    }

with open(logfile, "w") as f, open(csvfile, "w", newline="") as cf:
    def log(msg, quiet=False):
        if not quiet: print(msg)
        f.write(msg + "\n")

    writer = csv.writer(cf)
    writer.writerow([
        "kernel","N",
        "runtime_scalar_mean","runtime_scalar_std",
        "runtime_simd_mean","runtime_simd_std",
        "speedup",
        "gflops_scalar","gflops_simd",
        "cpe_scalar","cpe_simd",
        "validated","rtol","atol","abs_err","rel_err"
    ])

    log("=== Locality Sweep Experiment ===")
    log(f"Timestamp: {ts}")
    log(f"Warmup enabled: {WARMUP_CACHE}")
    log(f"CPU frequency used for CPE: {CPU_FREQ_HZ/1e9:.2f} GHz")

    # Compile scalar + SIMD versions
    log("\n[Compiling scalar baseline]")
    log(run_cmd(["g++","-O1","-fno-tree-vectorize","-std=c++17","-o",SCALAR_EXE,SRC_FILE]))
    log("\n[Compiling SIMD version]")
    log(run_cmd(["g++","-O3","-march=native","-ffast-math","-std=c++17","-o",SIMD_EXE,SRC_FILE]))

    # Broader sweep per kernel, covering cache boundaries
    kernel_sweeps = {}
    for kernel in FLOPS_PER_ELEM:
        thresholds = cache_thresholds(kernel)
        N_min = max(32, thresholds["L1"] // 8)   # start well below L1
        N_max = thresholds["L3"] * 8             # well past L3 into DRAM
        sweep = np.unique(
            np.logspace(np.log10(N_min), np.log10(N_max), num=40, dtype=int)
        )
        kernel_sweeps[kernel] = sweep

    for kernel in FLOPS_PER_ELEM:
        for N in kernel_sweeps[kernel]:
            log(f"\n--- {kernel}: N={N} ---")
            s_times, v_times, s_vals, v_vals = [], [], [], []

            num_runs = RUNS_PER_N
            warmup_extra = 1 if WARMUP_CACHE else 0

            for run in range(num_runs + warmup_extra):
                s_out = run_cmd([f"./{SCALAR_EXE}", str(N)])
                v_out = run_cmd([f"./{SIMD_EXE}", str(N)])

                if WARMUP_CACHE and run == 0:
                    continue  # discard warmup

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
                sp = t_s_mean / t_v_mean if t_v_mean > 0 else float("nan")

                # GFLOP/s
                flops = FLOPS_PER_ELEM[kernel] * N
                gflops_scalar = flops / (t_s_mean * 1e9) if t_s_mean > 0 else float("nan")
                gflops_simd   = flops / (t_v_mean * 1e9) if t_v_mean > 0 else float("nan")

                # CPE (cycles per element)
                cpe_scalar = (t_s_mean * CPU_FREQ_HZ / N) if (CPU_FREQ_HZ and N > 0) else float("nan")
                cpe_simd   = (t_v_mean * CPU_FREQ_HZ / N) if (CPU_FREQ_HZ and N > 0) else float("nan")

                # Validation
                s_val_mean = statistics.mean(s_vals) if s_vals else float("nan")
                v_val_mean = statistics.mean(v_vals) if v_vals else float("nan")
                validated, abs_err, rel_err = validate(s_val_mean, v_val_mean)

                writer.writerow([
                    kernel,N,
                    f"{t_s_mean:.6e}",f"{t_s_std:.6e}",
                    f"{t_v_mean:.6e}",f"{t_v_std:.6e}",
                    f"{sp:.3f}",
                    f"{gflops_scalar:.3f}",f"{gflops_simd:.3f}",
                    f"{cpe_scalar:.3f}",f"{cpe_simd:.3f}",
                    int(validated), RTOL, ATOL,
                    f"{abs_err:.3e}", f"{rel_err:.3e}"
                ])

                log(f"Runtime scalar={t_s_mean:.3e}s, SIMD={t_v_mean:.3e}s, speedup={sp:.2f}")
                log(f"GFLOP/s scalar={gflops_scalar:.2f}, SIMD={gflops_simd:.2f}")
                log(f"CPE scalar={cpe_scalar:.2f}, SIMD={cpe_simd:.2f}")
                log(f"Validation {kernel} N={N}: "
                    f"scalar_val={s_val_mean:.6e}, simd_val={v_val_mean:.6e}, "
                    f"validated={validated}, rel_err={rel_err:.2e}")

# === Plotting ===
df = pd.read_csv(csvfile)

for kernel in df["kernel"].unique():
    sub = df[df.kernel==kernel].copy()
    thresholds = cache_thresholds(kernel)

    # Error bars for GFLOP/s
    sub["gflops_scalar_err"] = sub["gflops_scalar"].astype(float) * (
        sub["runtime_scalar_std"].astype(float) / sub["runtime_scalar_mean"].astype(float)
    )
    sub["gflops_simd_err"] = sub["gflops_simd"].astype(float) * (
        sub["runtime_simd_std"].astype(float) / sub["runtime_simd_mean"].astype(float)
    )

    # Error bars for speedup
    rs_mean = sub["runtime_scalar_mean"].astype(float)
    rs_std  = sub["runtime_scalar_std"].astype(float)
    rv_mean = sub["runtime_simd_mean"].astype(float)
    rv_std  = sub["runtime_simd_std"].astype(float)
    sp = sub["speedup"].astype(float)
    sub["speedup_err"] = sp * ( (rs_std/rs_mean)**2 + (rv_std/rv_mean)**2 )**0.5

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
    plt.title(f"{kernel}: Runtime")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"{kernel}_runtime.png"))
    plt.close()

    # Speedup plot
    plt.figure()
    plt.errorbar(sub["N"].to_numpy(),
                 sp.to_numpy(),
                 yerr=sub["speedup_err"].to_numpy(),
                 marker="s", label="Speedup")
    for lvl, Nth in thresholds.items():
        plt.axvline(Nth, color="gray", linestyle="--", alpha=0.7)
        plt.text(Nth, plt.ylim()[1]*0.9, lvl, rotation=90,
                 va="bottom", ha="right", fontsize=8)
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("Speedup (Scalar / SIMD)")
    plt.title(f"{kernel}: Speedup")
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
    plt.title(f"{kernel}: Throughput")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"{kernel}_gflops.png"))
    plt.close()

    # CPE plot
    plt.figure()
    plt.errorbar(sub["N"].to_numpy(),
                 sub["cpe_scalar"].astype(float).to_numpy(),
                 yerr=(sub["cpe_scalar"].astype(float) *
                       sub["runtime_scalar_std"].astype(float) /
                       sub["runtime_scalar_mean"].astype(float)).to_numpy(),
                 marker="o", label="Scalar")
    plt.errorbar(sub["N"].to_numpy(),
                 sub["cpe_simd"].astype(float).to_numpy(),
                 yerr=(sub["cpe_simd"].astype(float) *
                       sub["runtime_simd_std"].astype(float) /
                       sub["runtime_simd_mean"].astype(float)).to_numpy(),
                 marker="o", label="SIMD")
    for lvl, Nth in thresholds.items():
        plt.axvline(Nth, color="gray", linestyle="--", alpha=0.7)
        plt.text(Nth, plt.ylim()[1]*0.9, lvl, rotation=90,
                 va="bottom", ha="right", fontsize=8)
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel("Cycles per Element (CPE)")
    plt.title(f"{kernel}: Cycles per Element")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"{kernel}_cpe.png"))
    plt.close()

print(f"Locality sweep log: {logfile}")
print(f"Locality sweep results: {csvfile}")
print(f"Locality sweep plots in {plotdir}/")
