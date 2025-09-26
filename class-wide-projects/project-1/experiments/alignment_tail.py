#!/usr/bin/env python3
import subprocess, datetime, os, csv, statistics, sys
import re as regex
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SRC_FILE = "kernels.cpp"
SCALAR_EXE = "kernels_scalar"
SIMD_EXE   = "kernels_simd"

# Warmup flag
WARMUP_CACHE = ("--warmup" in sys.argv)

LOG_DIR, CSV_DIR, PLOT_DIR = "logs", "results", "plots"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cache_state = "warm" if WARMUP_CACHE else "cold"
logfile = os.path.join(LOG_DIR, f"align_tail_{ts}_{cache_state}.log")
csvfile = os.path.join(CSV_DIR, f"align_tail_{ts}_{cache_state}.csv")
plotdir = os.path.join(PLOT_DIR, f"align_tail_{ts}_{cache_state}")
os.makedirs(plotdir, exist_ok=True)

RUNS_PER_CASE = 5
RTOL, ATOL = 1e-5, 1e-7

FLOPS_PER_ELEM   = {"SAXPY":2,"Dot":2,"ElemMul":1,"Stencil":3}
KERNELS = list(FLOPS_PER_ELEM.keys())

# CPU frequency (for info only)
def detect_cpu_freq():
    try:
        out = subprocess.run(["lscpu"], check=True, stdout=subprocess.PIPE, text=True).stdout
        m = regex.search(r"CPU max MHz:\s*([\d.]+)", out) or regex.search(r"CPU MHz:\s*([\d.]+)", out)
        if m: return float(m.group(1)) * 1e6
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "cpu MHz" in line:
                    return float(line.split(":")[1].strip()) * 1e6
    except Exception:
        pass
    return 3.0e9

CPU_FREQ_HZ = detect_cpu_freq()

def run_cmd(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout

def parse_output(out_text: str):
    """Parse stdout from kernels_* executables."""
    results, values = {}, {}
    for line in str(out_text).splitlines():
        line = line.strip()
        if not line or "time" not in line or "=" not in line:
            continue
        m = regex.match(r"(\w+)\s+time\s*=\s*([0-9.eE+-]+).*=\s*([0-9.eE+-]+)", line)
        if m:
            kernel = m.group(1)
            t = float(m.group(2))
            val = float(m.group(3))
            results[kernel] = t
            values[kernel]  = val
    return results, values

def validate(ref_val, test_val):
    abs_err = abs(test_val - ref_val)
    rel_err = abs_err / max(abs(ref_val), 1e-30)
    ok = (abs_err <= ATOL) or (rel_err <= RTOL)
    return ok, abs_err, rel_err

# Alignment offsets (in elements)
ALIGN_CASES = [("aligned", 0), ("misaligned", 1)]

# Tail cases
def build_N_variants(base_N):
    n16 = (base_N // 16) * 16
    n8  = (base_N // 8) * 8
    return {
        "no_tail_16x": n16 if n16 > 0 else 16,
        "no_tail_8x":  n8 if (n8 % 16) != 0 else n8 + 8,
        "tail":        n8 + 3,
    }

BASE_N = 200000  # L2/L3-ish region

with open(logfile, "w") as lf, open(csvfile, "w", newline="") as cf:
    def log(msg, quiet=False):
        if not quiet: print(msg)
        lf.write(msg + "\n")

    writer = csv.writer(cf)
    writer.writerow([
        "kernel","N","tail_kind","alignment","offset",
        "runtime_scalar_mean","runtime_scalar_std",
        "runtime_simd_mean","runtime_simd_std",
        "speedup","gflops_scalar","gflops_simd",
        "runtime_scalar_median","runtime_scalar_var",
        "runtime_simd_median","runtime_simd_var",
        "speedup_median","speedup_var",
        "gflops_scalar_median","gflops_scalar_var",
        "gflops_simd_median","gflops_simd_var",
        "validated","abs_err","rel_err"
    ])

    log("=== Alignment & Tail Experiment ===")
    log(f"Warmup: {WARMUP_CACHE}, CPU freq {CPU_FREQ_HZ/1e9:.2f} GHz")

    # Compile
    log("[Compiling scalar]")
    log(run_cmd(["g++","-O1","-fno-tree-vectorize","-std=c++17","-o",SCALAR_EXE,SRC_FILE]))
    log("[Compiling SIMD]")
    log(run_cmd(["g++","-O3","-march=native","-ffast-math","-std=c++17","-o",SIMD_EXE,SRC_FILE]))

    for kernel in KERNELS:
        Ns = build_N_variants(BASE_N)
        for tail_kind, N in Ns.items():
            for align_name, offset in ALIGN_CASES:
                log(f"\n--- {kernel}, N={N}, {tail_kind}, {align_name} ---")
                s_times, v_times, s_vals, v_vals = [], [], [], []
                runs = RUNS_PER_CASE + (1 if WARMUP_CACHE else 0)

                for r in range(runs):
                    s_out = run_cmd([f"./{SCALAR_EXE}", str(N), str(offset)])
                    v_out = run_cmd([f"./{SIMD_EXE}",   str(N), str(offset)])
                    if WARMUP_CACHE and r == 0:
                        continue
                    s_res, s_val = parse_output(s_out)
                    v_res, v_val = parse_output(v_out)
                    if kernel in s_res:
                        s_times.append(s_res[kernel])
                        s_vals.append(s_val[kernel])
                    if kernel in v_res:
                        v_times.append(v_res[kernel])
                        v_vals.append(v_val[kernel])

                if s_times and v_times:
                    t_s = statistics.mean(s_times)
                    t_v = statistics.mean(v_times)
                    sd_s = statistics.stdev(s_times) if len(s_times) > 1 else 0.0
                    sd_v = statistics.stdev(v_times) if len(v_times) > 1 else 0.0

                    t_s_median = statistics.median(s_times)
                    t_s_var = statistics.variance(s_times) if len(s_times) > 1 else 0.0
                    t_v_median = statistics.median(v_times)
                    t_v_var = statistics.variance(v_times) if len(v_times) > 1 else 0.0

                    sp = t_s / t_v if t_v > 0 else float("nan")
                    flops = FLOPS_PER_ELEM[kernel] * N
                    gflops_s = flops / (t_s * 1e9) if t_s > 0 else float("nan")
                    gflops_v = flops / (t_v * 1e9) if t_v > 0 else float("nan")
                    # approximate error bars for GFLOP/s by propagating relative runtime error
                    gflops_s_err = gflops_s * (sd_s / t_s if t_s > 0 else 0.0)
                    gflops_v_err = gflops_v * (sd_v / t_v if t_v > 0 else 0.0)

                    gflops_s_runs = [flops / (t * 1e9) for t in s_times if t > 0]
                    gflops_v_runs = [flops / (t * 1e9) for t in v_times if t > 0]
                    gflops_s_median = statistics.median(gflops_s_runs) if gflops_s_runs else float("nan")
                    gflops_s_var = statistics.variance(gflops_s_runs) if len(gflops_s_runs) > 1 else 0.0
                    gflops_v_median = statistics.median(gflops_v_runs) if gflops_v_runs else float("nan")
                    gflops_v_var = statistics.variance(gflops_v_runs) if len(gflops_v_runs) > 1 else 0.0

                    s_val_m = statistics.mean(s_vals) if s_vals else float("nan")
                    v_val_m = statistics.mean(v_vals) if v_vals else float("nan")
                    ok, abs_err, rel_err = validate(s_val_m, v_val_m)

                    speedup_runs = [s/v for s, v in zip(s_times, v_times) if v > 0]
                    sp_median = statistics.median(speedup_runs) if speedup_runs else float("nan")
                    sp_var = statistics.variance(speedup_runs) if len(speedup_runs) > 1 else 0.0

                    writer.writerow([
                        kernel, N, tail_kind, align_name, offset,
                        t_s, sd_s, t_v, sd_v,
                        sp, gflops_s, gflops_v,
                        t_s_median, t_s_var, t_v_median, t_v_var,
                        sp_median, sp_var,
                        gflops_s_median, gflops_s_var,
                        gflops_v_median, gflops_v_var,
                        int(ok), abs_err, rel_err
                    ])

                    log(f"{kernel} speedup={sp:.2f}, GFLOP/s SIMD={gflops_v:.2f}")

# === Plotting ===
df = pd.read_csv(csvfile)
for kernel in df.kernel.unique():
    sub = df[df.kernel == kernel].copy()
    tail_order = ["no_tail_16x","no_tail_8x","tail"]
    sub["tail_kind"] = pd.Categorical(sub["tail_kind"], categories=tail_order, ordered=True)
    sub.sort_values(["tail_kind","alignment"], inplace=True)

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

    # GFLOP/s
    plt.figure()
    for align in ["aligned","misaligned"]:
        ss = sub[sub.alignment == align]
        x = np.arange(len(ss.tail_kind))
        plt.errorbar(x, ss["gflops_simd"], 
                     yerr=ss["runtime_simd_std"] / ss["runtime_simd_mean"] * ss["gflops_simd"],
                     marker="o", label=f"SIMD-{align}", capsize=4)
        plt.errorbar(x, ss["gflops_scalar"], 
                     yerr=ss["runtime_scalar_std"] / ss["runtime_scalar_mean"] * ss["gflops_scalar"],
                     marker="s", label=f"Scalar-{align}", capsize=4)
    plt.xticks(range(len(tail_order)), tail_order)
    plt.ylabel("GFLOP/s"); plt.title(f"{kernel} throughput (GFLOP/s)")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"{kernel}_gflops.png")); plt.close()

    # Runtime
    plt.figure()
    for align in ["aligned","misaligned"]:
        ss = sub[sub.alignment == align]
        x = np.arange(len(ss.tail_kind))
        plt.errorbar(x, ss["runtime_simd_mean"], yerr=ss["runtime_simd_std"],
                     marker="o", label=f"SIMD-{align}", capsize=4)
        plt.errorbar(x, ss["runtime_scalar_mean"], yerr=ss["runtime_scalar_std"],
                     marker="s", label=f"Scalar-{align}", capsize=4)
    plt.xticks(range(len(tail_order)), tail_order)
    plt.ylabel("Runtime (s)"); plt.title(f"{kernel} runtime")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"{kernel}_runtime.png")); plt.close()

    # Speedup (no error bars because scalar+SIMD runs are correlated)
    plt.figure()
    for align in ["aligned","misaligned"]:
        ss = sub[sub.alignment == align]
        plt.plot(ss["tail_kind"].to_numpy(), ss["speedup"].to_numpy(),
                 marker="o", label=align)
    plt.axhline(1.0, color="gray", linestyle="--")
    plt.ylabel("Speedup (scalar / SIMD)"); plt.title(f"{kernel} speedup")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plotdir, f"{kernel}_speedup.png")); plt.close()

print(f"Alignment/tail log: {logfile}")
print(f"Alignment/tail results: {csvfile}")
print(f"Alignment/tail plots in {plotdir}/")
