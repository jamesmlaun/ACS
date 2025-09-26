#!/usr/bin/env python3
import subprocess, datetime, os, csv, re, statistics, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SRC_FILE = "kernels.cpp"
SCALAR_EXE = "kernels_scalar"
SIMD_EXE   = "kernels_simd"

WARMUP = ("--warmup" in sys.argv)

LOG_DIR, CSV_DIR, PLOT_DIR = "logs", "results", "plots"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cache_state = "warm" if WARMUP else "cold"
logfile = os.path.join(LOG_DIR, f"datatype_{ts}_{cache_state}.log")
csvfile = os.path.join(CSV_DIR, f"datatype_{ts}_{cache_state}.csv")
plotdir = os.path.join(PLOT_DIR, f"datatype_{ts}_{cache_state}")
os.makedirs(plotdir, exist_ok=True)

RUNS = 3 + (1 if WARMUP else 0)

ARRAYS_PER_KERNEL = {"SAXPY":2,"Dot":2,"ElemMul":3,"Stencil":2}
FLOPS_PER_ELEM = {"SAXPY":2,"Dot":2,"ElemMul":1,"Stencil":3}
ELEM_SIZE = {"float32":4, "float64":8}

# Cache sizes
L1_BYTES, L2_BYTES, L3_BYTES = 48*1024, 1280*1024, 24*1024*1024

def run(cmd): 
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout

def parse(out):
    res, val = {}, {}
    for line in out.splitlines():
        m = re.match(r"(\w+)\s+time\s*=\s*([0-9.eE+-]+).*=\s*([0-9.eE+-]+)", line.strip())
        if m:
            res[m.group(1)] = float(m.group(2))
            val[m.group(1)] = float(m.group(3))
    return res, val

def n_midpoints(arrays, elem_size):
    N_L1   = (L1_BYTES // 2) // (arrays * elem_size)
    N_L2   = ((L1_BYTES + L2_BYTES) // 2) // (arrays * elem_size)
    N_L3   = ((L2_BYTES + L3_BYTES) // 2) // (arrays * elem_size)
    N_DRAM = ((L3_BYTES*2 + L3_BYTES*8)//2) // (arrays * elem_size)
    return [N_L1, N_L2, N_L3, N_DRAM]

# Compile
print("=== Data Type Comparison Experiment ===")
print("[Compiling scalar]")
print(run(["g++","-O1","-fno-tree-vectorize","-std=c++17","-o",SCALAR_EXE,SRC_FILE]))
print("[Compiling SIMD]")
print(run(["g++","-O3","-march=native","-ffast-math","-std=c++17","-o",SIMD_EXE,SRC_FILE]))

with open(csvfile,"w",newline="") as cf, open(logfile,"w") as lf:
    wr = csv.writer(cf)
    wr.writerow(["kernel","dtype","N",
                 "runtime_scalar_mean","runtime_scalar_std",
                 "runtime_simd_mean","runtime_simd_std",
                 "speedup","gflops_scalar","gflops_simd"])

    def log(msg): 
        print(msg); lf.write(msg+"\n")

    kernels = list(FLOPS_PER_ELEM.keys())
    dtypes = ["float32","float64"]

    for kernel in kernels:
        for dtype in dtypes:
            elem_size = ELEM_SIZE[dtype]
            arrays = ARRAYS_PER_KERNEL[kernel]
            sweep = n_midpoints(arrays, elem_size)

            for N in sweep:
                log(f"--- {kernel}, {dtype}, N={N} ---")
                s_times, v_times = [], []
                for r in range(RUNS):
                    so = run([f"./{SCALAR_EXE}", str(N)])
                    vo = run([f"./{SIMD_EXE}", str(N)])
                    if WARMUP and r == 0: 
                        continue
                    sr,_ = parse(so); vr,_ = parse(vo)
                    if kernel in sr: s_times.append(sr[kernel])
                    if kernel in vr: v_times.append(vr[kernel])

                if not s_times or not v_times: 
                    continue
                tS, tV = statistics.mean(s_times), statistics.mean(v_times)
                sdS = statistics.stdev(s_times) if len(s_times)>1 else 0.0
                sdV = statistics.stdev(v_times) if len(v_times)>1 else 0.0
                sp = tS/tV if tV>0 else float("nan")
                flops = FLOPS_PER_ELEM[kernel] * N
                gS = flops/(tS*1e9); gV = flops/(tV*1e9)
                wr.writerow([kernel,dtype,N,tS,sdS,tV,sdV,sp,gS,gV])
                log(f"{kernel} {dtype}: speedup={sp:.2f}, GFLOP/s SIMD={gV:.2f}")

# === Plotting ===
df = pd.read_csv(csvfile)

for kernel in df.kernel.unique():
    for metric, ylabel in [
        ("runtime","Runtime (s)"),
        ("gflops","GFLOP/s"),
        ("speedup","Speedup (Scalar/SIMD)")
    ]:
        plt.figure()

        for dtype in ["float32","float64"]:
            sub = df[(df.kernel==kernel) & (df.dtype==dtype)].copy()
            if sub.empty:
                continue

            x = sub["N"].to_numpy()

            if metric == "runtime":
                y = sub[metric].astype(float).to_numpy()
                yerr = sub["runtime_simd_std"].astype(float).to_numpy()
                plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=f"{dtype} SIMD")
                # scalar
                ys = sub["runtime_scalar_mean"].astype(float).to_numpy()
                ys_err = sub["runtime_scalar_std"].astype(float).to_numpy()
                plt.errorbar(x, ys, yerr=ys_err, marker="s", capsize=4, label=f"{dtype} Scalar")

            elif metric == "gflops":
                g_s = sub["gflops_scalar"].astype(float).to_numpy()
                g_v = sub["gflops_simd"].astype(float).to_numpy()
                g_s_err = g_s * (sub["runtime_scalar_std"].astype(float) / sub["runtime_scalar_mean"].astype(float)).to_numpy()
                g_v_err = g_v * (sub["runtime_simd_std"].astype(float) / sub["runtime_simd_mean"].astype(float)).to_numpy()
                plt.errorbar(x, g_v, yerr=g_v_err, marker="o", capsize=4, label=f"{dtype} SIMD")
                plt.errorbar(x, g_s, yerr=g_s_err, marker="s", capsize=4, label=f"{dtype} Scalar")

            elif metric == "speedup":
                sp = sub["speedup"].astype(float).to_numpy()
                rs_mean = sub["runtime_scalar_mean"].astype(float)
                rs_std  = sub["runtime_scalar_std"].astype(float)
                rv_mean = sub["runtime_simd_mean"].astype(float)
                rv_std  = sub["runtime_simd_std"].astype(float)
                sp_err = sp * np.sqrt((rs_std/rs_mean)**2 + (rv_std/rv_mean)**2).to_numpy()
                plt.errorbar(x, sp, yerr=sp_err, marker="o", capsize=4, label=dtype)

        plt.xscale("log")
        if metric!="speedup":
            plt.yscale("log")
        plt.xlabel("N")
        plt.ylabel(ylabel)
        plt.title(f"{kernel}: {ylabel} by data type")
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join(plotdir, f"{kernel}_{metric}.png"))
        plt.close()

print(f"Datatype log: {logfile}")
print(f"Datatype results: {csvfile}")
print(f"Datatype plots in {plotdir}/")
