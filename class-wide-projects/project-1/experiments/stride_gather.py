#!/usr/bin/env python3
import subprocess, datetime, os, csv, statistics, sys, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SRC_FILE = "kernels.cpp"
SCALAR_EXE = "kernels_scalar"
SIMD_EXE   = "kernels_simd"

# Warmup flag
WARMUP = ("--warmup" in sys.argv)

LOG_DIR, CSV_DIR, PLOT_DIR = "logs", "results", "plots"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cache_state = "warm" if WARMUP else "cold"
logfile = os.path.join(LOG_DIR, f"stride_{ts}_{cache_state}.log")
csvfile = os.path.join(CSV_DIR, f"stride_{ts}_{cache_state}.csv")
plotdir = os.path.join(PLOT_DIR, f"stride_{ts}_{cache_state}")
os.makedirs(plotdir, exist_ok=True)

RUNS = 3
FLOPS_PER_ELEM = {"SAXPY": 2, "Dot": 2, "ElemMul": 1, "Stencil": 3}

def run_cmd(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout

def parse_output(out):
    res, val = {}, {}
    for line in out.splitlines():
        m = re.match(r"(\w+) time = ([0-9.eE+-]+) s.*= ([0-9.eE+-]+)", line)
        if m:
            res[m.group(1)] = float(m.group(2))
            val[m.group(1)] = float(m.group(3))
    return res, val

stride_choices = [1, 2, 8, 32]
gather_cfgs = [("gather-blocked", 1, 0.5), ("gather-random", 2, 0.5)]
kernels = ["SAXPY", "Dot", "ElemMul", "Stencil"]

with open(logfile, "w") as f, open(csvfile, "w", newline="") as cf:
    def log(msg, quiet=False):
        if not quiet: print(msg)
        f.write(msg + "\n")

    writer = csv.writer(cf)
    writer.writerow([
        "pattern","op","label","N","stride","gather_mode","frac",
        "runtime_scalar_mean","runtime_scalar_std",
        "runtime_simd_mean","runtime_simd_std",
        "speedup","useful_elems"
    ])

    log("=== Stride / Gather Experiment ===")
    log(f"Timestamp: {ts}")
    log(f"Warmup enabled: {WARMUP}")

    # Compile
    log("\n[Compiling scalar baseline]")
    log(run_cmd(["g++","-O1","-fno-tree-vectorize","-std=c++17","-o",SCALAR_EXE,SRC_FILE]))
    log("\n[Compiling SIMD version]")
    log(run_cmd(["g++","-O3","-march=native","-ffast-math","-std=c++17","-o",SIMD_EXE,SRC_FILE]))

    N = int(32e6)

    # Strided
    for i, s in enumerate(stride_choices, 1):
        for op in kernels:
            label = "unit-stride" if s == 1 else f"stride-{s}"
            log(f"\n[{i}/{len(stride_choices)}] Kernel={op}, Mode=Stride, stride={s} ({label})")
            s_times, v_times = [], []
            for r in range(RUNS + (1 if WARMUP else 0)):
                s_out = run_cmd([f"./{SCALAR_EXE}", str(N), "0", "stride", str(s)])
                v_out = run_cmd([f"./{SIMD_EXE}", str(N), "0", "stride", str(s)])

                if WARMUP and r == 0:
                    log("  Skipping warmup run")
                    continue

                sr, _ = parse_output(s_out)
                vr, _ = parse_output(v_out)

                if op in sr and op in vr:
                    s_times.append(sr[op]); v_times.append(vr[op])

                log(f"  Run {r+1}: scalar={sr.get(op,'-')}, simd={vr.get(op,'-')}")

            if not s_times or not v_times: continue
            tS, tV = statistics.mean(s_times), statistics.mean(v_times)
            sdS = statistics.stdev(s_times) if len(s_times)>1 else 0.0
            sdV = statistics.stdev(v_times) if len(v_times)>1 else 0.0

            N_used = (N + s - 1) // s
            writer.writerow(["strided",op,label,N,s,0,1.0,
                             tS,sdS,tV,sdV,tS/tV,N_used])

    # Gather
    for j, (name, mode, frac) in enumerate(gather_cfgs, 1):
        for op in kernels:
            log(f"\n[{j}/{len(gather_cfgs)}] Kernel={op}, Mode=Gather-{name}, frac={frac}")
            s_times, v_times = [], []
            for r in range(RUNS + (1 if WARMUP else 0)):
                s_out = run_cmd([f"./{SCALAR_EXE}", str(N), "0", "gather", str(mode), str(frac)])
                v_out = run_cmd([f"./{SIMD_EXE}", str(N), "0", "gather", str(mode), str(frac)])

                if WARMUP and r == 0:
                    log("  Skipping warmup run")
                    continue

                sr, _ = parse_output(s_out)
                vr, _ = parse_output(v_out)

                if op in sr and op in vr:
                    s_times.append(sr[op]); v_times.append(vr[op])

                log(f"  Run {r+1}: scalar={sr.get(op,'-')}, simd={vr.get(op,'-')}")

            if not s_times or not v_times: continue
            tS, tV = statistics.mean(s_times), statistics.mean(v_times)
            sdS = statistics.stdev(s_times) if len(s_times)>1 else 0.0
            sdV = statistics.stdev(v_times) if len(v_times)>1 else 0.0

            N_used = int(N * frac)
            writer.writerow([f"gather:{name}",op,name,N,1,mode,frac,
                             tS,sdS,tV,sdV,tS/tV,N_used])

print(f"\nStride/gather log: {logfile}")
print(f"Stride/gather results: {csvfile}")
print(f"Stride/gather plots in {plotdir}/")

# === Plotting ===
df = pd.read_csv(csvfile)
labels = ["unit-stride","stride-2","stride-8","stride-32","gather-blocked","gather-random"]

def plot_consolidated(kernel):
    sub = df[df["op"]==kernel].copy()
    if sub.empty: return
    sub["label"] = pd.Categorical(sub["label"], categories=labels, ordered=True)
    sub.sort_values("label", inplace=True)
    x = np.arange(len(sub))

    flops = FLOPS_PER_ELEM[kernel] * sub["useful_elems"].astype(float)
    sub["gflops_scalar"] = flops / (sub["runtime_scalar_mean"].astype(float)*1e9)
    sub["gflops_simd"]   = flops / (sub["runtime_simd_mean"].astype(float)*1e9)
    sub["gflops_scalar_err"] = sub["gflops_scalar"]*(sub["runtime_scalar_std"].astype(float)/sub["runtime_scalar_mean"].astype(float))
    sub["gflops_simd_err"]   = sub["gflops_simd"]*(sub["runtime_simd_std"].astype(float)/sub["runtime_simd_mean"].astype(float))

    # GFLOP/s
    plt.figure()
    plt.errorbar(x, sub["gflops_simd"].to_numpy(), yerr=sub["gflops_simd_err"].to_numpy(),
                 marker="o", label="SIMD", capsize=4)
    plt.errorbar(x, sub["gflops_scalar"].to_numpy(), yerr=sub["gflops_scalar_err"].to_numpy(),
                 marker="s", label="Scalar", capsize=4)
    plt.xticks(x, sub["label"], rotation=45, ha='right')
    plt.ylabel("GFLOP/s"); plt.title(f"{kernel}: Throughput")
    plt.tight_layout()
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(plotdir, f"{kernel.lower()}_gflops.png")); plt.close()

    # Runtime
    plt.figure()
    plt.errorbar(x, sub["runtime_simd_mean"].to_numpy(), yerr=sub["runtime_simd_std"].to_numpy(),
                 marker="o", label="SIMD", capsize=4)
    plt.errorbar(x, sub["runtime_scalar_mean"].to_numpy(), yerr=sub["runtime_scalar_std"].to_numpy(),
                 marker="s", label="Scalar", capsize=4)
    plt.xticks(x, sub["label"], rotation=45, ha='right')
    plt.ylabel("Runtime (s)"); plt.title(f"{kernel}: Runtime")
    plt.tight_layout()
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(plotdir, f"{kernel.lower()}_runtime.png")); plt.close()

    # Speedup
    plt.figure()
    plt.plot(x, sub["speedup"].to_numpy(), marker="o", label="Speedup")
    plt.axhline(1.0, color="gray", linestyle="--")
    plt.xticks(x, sub["label"], rotation=45, ha='right')
    plt.ylabel("Speedup (scalar / SIMD)"); plt.title(f"{kernel}: Speedup")
    plt.tight_layout()
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(plotdir, f"{kernel.lower()}_speedup.png")); plt.close()

for kernel in kernels:
    plot_consolidated(kernel)
