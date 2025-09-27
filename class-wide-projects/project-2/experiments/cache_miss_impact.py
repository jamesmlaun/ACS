"""
Cache-Miss Impact (stride only, constant work)
- Compiles saxpy.cpp -> saxpy
- Fixes N, sweeps stride values
- Each run executes exactly N*ITERS updates (constant work)
- Collects runtime + perf counters
- Outputs raw CSV, summary CSV
- Produces a single plot: Runtime vs Miss Rate (dots connected)
"""

import subprocess, csv, time, random, statistics, re
from pathlib import Path
import matplotlib.pyplot as plt
from experiments import utils

# ---------- Config ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC = ROOT_DIR / "saxpy.cpp"
BIN = ROOT_DIR / "saxpy"

# Fixed problem size (~64 MB footprint, 16M floats)
N = 16 * 1024 * 1024
A_SCALAR = 2.0
ITERS = 4  # repeat count to keep runtime measurable

# Sweep strides up to 2^5 = 32 elements (odd strides to avoid aliasing)
STRIDES = [1, 3, 5, 9, 17, 31]

RUNTIME_RE = re.compile(r"runtime_ms=([\d.]+)")

# ---------- Helpers ----------
def compile_kernel(log):
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source file: {SRC}")
    cmd = ["g++", "-O2", "-march=native", str(SRC), "-o", str(BIN)]
    log.write("[COMPILE] " + " ".join(cmd) + "\n")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout: log.write("[COMPILE-STDOUT]\n" + res.stdout + "\n")
    if res.stderr: log.write("[COMPILE-STDERR]\n" + res.stderr + "\n")
    if res.returncode != 0:
        raise RuntimeError("Compilation failed (see log).")

def parse_perf_output(text: str):
    refs, misses = None, None
    for line in utils.filter_and_log(text):
        if "cache-references" in line:
            try: refs = int(line.strip().split()[0].replace(",", ""))
            except: pass
        elif "cache-misses" in line:
            try: misses = int(line.strip().split()[0].replace(",", ""))
            except: pass
    return refs, misses

# ---------- Main Runner ----------
def run(reps=3, warmup=True, randomize=False):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = Path("logs")
    results_dir = Path("results") / f"cache_miss_impact_{ts}"
    figures_dir = Path("figures") / f"cache_miss_impact_{ts}"
    for d in (logs_dir, results_dir, figures_dir): d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"cache_miss_impact_{ts}.log"
    raw_csv = results_dir / "raw_data.csv"
    summary_csv = results_dir / "summary.csv"
    corr_plot = figures_dir / "runtime_vs_missrate.png"

    print(f"[INFO] Cache-Miss Impact (stride only, constant work)")

    raw_data = {s: [] for s in STRIDES}

    with open(log_file,"w") as log, open(raw_csv,"w",newline="") as raw_out:
        writer = csv.writer(raw_out)
        writer.writerow(["rep","stride","iters","runtime_ms","cache_refs","cache_misses","miss_rate"])

        utils.log_environment(log, "perf")

        # Compile binary
        compile_kernel(log)

        # Warmup (stride=1 baseline)
        if warmup:
            cmd = ["perf","stat","-e","cache-references,cache-misses","taskset","-c","0",
                   str(BIN),str(N),str(A_SCALAR),"stride",str(ITERS),"1"]
            log.write("[WARMUP] " + " ".join(cmd) + "\n")
            subprocess.run(cmd,capture_output=True,text=True)

        # Main reps
        for rep in range(1,reps+1):
            order = list(STRIDES)
            if randomize: random.shuffle(order)
            for stride in order:
                print(f"[INFO] rep={rep}, stride={stride}")
                args = [str(N), str(A_SCALAR), "stride", str(ITERS), str(stride)]
                cmd = ["perf","stat","-e","cache-references,cache-misses","taskset","-c","0",str(BIN)] + args
                log.write(f"[RUN] rep={rep}, stride={stride}\nCommand: " + " ".join(cmd) + "\n")
                res = subprocess.run(cmd,capture_output=True,text=True)
                out, err = res.stdout, res.stderr
                if out: log.write(out+"\n")
                if err: log.write("[STDERR]\n"+err+"\n")

                # Parse runtime
                m = RUNTIME_RE.search(out)
                runtime = float(m.group(1)) if m else None
                refs, misses = parse_perf_output(err)
                miss_rate = (misses/refs) if (refs and misses is not None) else None

                if runtime is not None and miss_rate is not None:
                    raw_data[stride].append((runtime,refs,misses,miss_rate))
                    writer.writerow([rep,stride,ITERS,f"{runtime:.4f}",refs,misses,f"{miss_rate:.6f}"])

    # Summaries
    summary_rows = []
    for stride,vals in raw_data.items():
        if not vals: continue
        runtimes=[v[0] for v in vals]; missrates=[v[3] for v in vals]
        rt_mean=statistics.mean(runtimes); rt_std=statistics.stdev(runtimes) if len(runtimes)>1 else 0.0
        mr_mean=statistics.mean(missrates); mr_std=statistics.stdev(missrates) if len(missrates)>1 else 0.0
        summary_rows.append((stride,rt_mean,rt_std,mr_mean,mr_std))

    with open(summary_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["stride","runtime_mean_ms","runtime_std_ms","missrate_mean","missrate_std"])
        for row in summary_rows:
            w.writerow([row[0],f"{row[1]:.4f}",f"{row[2]:.4f}",f"{row[3]:.6f}",f"{row[4]:.6f}"])

    # Plot: Runtime vs Miss Rate
    plt.figure(figsize=(8,6))
    summary_rows.sort(key=lambda r:int(r[0]))
    mr=[r[3] for r in summary_rows]; rt=[r[1] for r in summary_rows]
    plt.plot(mr,rt,"o-",label="stride")
    plt.xlabel("Miss Rate"); plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Miss Rate (constant work, stride sweep)")
    plt.grid(True,linestyle="--",alpha=0.7)
    plt.legend()
    plt.savefig(corr_plot,dpi=300,bbox_inches="tight"); plt.close()

    print(f"[DONE] Results:\n - {raw_csv}\n - {summary_csv}\n - {corr_plot}\n - {log_file}")
