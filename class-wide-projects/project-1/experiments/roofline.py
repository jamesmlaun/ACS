#!/usr/bin/env python3
import subprocess, datetime, os, csv, sys, re, statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SRC_FILE = "kernels.cpp"
SCALAR_EXE = "kernels_scalar"
SIMD_EXE   = "kernels_simd"

# Warmup flag
WARMUP = ("--warmup" in sys.argv)

# Directories
LOG_DIR, CSV_DIR, PLOT_DIR = "logs", "results", "plots"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cache_state = "warm" if WARMUP else "cold"
logfile = os.path.join(LOG_DIR, f"roofline_{ts}_{cache_state}.log")
csvfile = os.path.join(CSV_DIR, f"roofline_{ts}_{cache_state}.csv")
plotdir = os.path.join(PLOT_DIR, f"roofline_{ts}_{cache_state}")
os.makedirs(plotdir, exist_ok=True)

RUNS = 3 + (1 if WARMUP else 0)

# Kernel info
ARRAYS_PER_KERNEL = {"SAXPY":2,"Dot":2,"ElemMul":3,"Stencil":2}
FLOPS_PER_ELEM    = {"SAXPY":2,"Dot":2,"ElemMul":1,"Stencil":3}
BYTES_PER_ELEM    = {"SAXPY":12,"Dot":8,"ElemMul":12,"Stencil":16}  # approximate DRAM traffic

# Hardware constants
PEAK_BW = 40.0  # GB/s, from Project 2
PEAK_FLOPS = 80.0  # GFLOP/s, 2.5 GHz × 32 FLOPs/cycle (AVX2 FMA per core)

def run_cmd(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout

def parse_output(out):
    res = {}
    for line in out.splitlines():
        m = re.match(r"(\w+)\s+time\s*=\s*([0-9.eE+-]+)", line.strip())
        if m:
            res[m.group(1)] = float(m.group(2))
    return res

print("=== Roofline Experiment ===")
print(f"Warmup: {WARMUP}")

with open(logfile, "w") as lf, open(csvfile, "w", newline="") as cf:
    def log(msg): print(msg); lf.write(msg+"\n")

    wr = csv.writer(cf)
    wr.writerow(["kernel","N","runtime_scalar","runtime_simd","gflops_scalar","gflops_simd","AI"])

    # Compile
    log("[Compiling scalar]")
    log(run_cmd(["g++","-O1","-fno-tree-vectorize","-std=c++17","-o",SCALAR_EXE,SRC_FILE]))
    log("[Compiling SIMD]")
    log(run_cmd(["g++","-O3","-march=native","-ffast-math","-std=c++17","-o",SIMD_EXE,SRC_FILE]))

    # DRAM-resident size: pick a large N (~32M elements)
    N = int(32e6)

    for kernel in ["SAXPY","Dot","Stencil"]:
        log(f"\n--- {kernel}, N={N} ---")
        s_times, v_times = [], []

        for r in range(RUNS):
            s_out = run_cmd([f"./{SCALAR_EXE}", str(N)])
            v_out = run_cmd([f"./{SIMD_EXE}", str(N)])
            if WARMUP and r == 0:
                log("  Skipping warmup run")
                continue
            sr = parse_output(s_out)
            vr = parse_output(v_out)
            if kernel in sr: s_times.append(sr[kernel])
            if kernel in vr: v_times.append(vr[kernel])

        if not s_times or not v_times:
            continue

        tS, tV = statistics.mean(s_times), statistics.mean(v_times)
        flops = FLOPS_PER_ELEM[kernel] * N
        gS = flops/(tS*1e9)
        gV = flops/(tV*1e9)

        # Arithmetic intensity (FLOPs/byte)
        AI = FLOPS_PER_ELEM[kernel] / BYTES_PER_ELEM[kernel]

        wr.writerow([kernel,N,tS,tV,gS,gV,AI])
        log(f"{kernel}: scalar={gS:.2f} GF/s, SIMD={gV:.2f} GF/s, AI={AI:.3f}")

# === Plotting ===
df = pd.read_csv(csvfile)

# Roofline curve
ai = np.logspace(-3,3,200)
perf_mem = PEAK_BW * ai
perf_comp = np.full_like(ai, PEAK_FLOPS)
roofline = np.minimum(perf_mem, perf_comp)

plt.figure(figsize=(7,5))
plt.loglog(ai, roofline, linewidth=2, label="Roofline")
plt.loglog(ai, perf_mem, linestyle="--", color="gray", label="Memory bound (40 GB/s)")
plt.loglog(ai, perf_comp, linestyle="--", color="gray", label="Compute bound (80 GF/s)")

# Plot kernel points
for _, row in df.iterrows():
    plt.scatter(row["AI"], row["gflops_simd"], s=60, label=f"{row['kernel']} SIMD ({row['gflops_simd']:.1f} GF/s)")
    plt.scatter(row["AI"], row["gflops_scalar"], s=60, marker="x", label=f"{row['kernel']} Scalar")

# Ridge point
ridge_ai = PEAK_FLOPS/PEAK_BW
plt.scatter([ridge_ai],[PEAK_FLOPS],s=80,color="red")
plt.annotate(f"Ridge AI≈{ridge_ai:.2f}", xy=(ridge_ai,PEAK_FLOPS),
             xytext=(ridge_ai*1.5,PEAK_FLOPS/2), arrowprops=dict(arrowstyle="->"))

plt.title("Roofline Model (i7-11850H, 1 core, FP32)")
plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Attainable Performance (GFLOP/s)")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plotdir,"roofline.png"),dpi=180)
plt.close()

print(f"Roofline log: {logfile}")
print(f"Roofline results: {csvfile}")
print(f"Roofline plot in {plotdir}/")
