"""
Cache-Miss Impact (sequential SAXPY)
- Compiles saxpy.cpp -> saxpy
- Sweeps working-set sizes to vary cache miss rate
- Collects runtime + cache counters
- Outputs raw CSV, summary CSV, and plots:
    1. Runtime vs problem size
    2. Miss rate vs problem size
    3. Runtime vs miss rate
"""

import subprocess, csv, time, random, statistics, re
from pathlib import Path
import matplotlib.pyplot as plt
from experiments import utils

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC = ROOT_DIR / "saxpy.cpp"
BIN = ROOT_DIR / "saxpy"

SIZES = [
    2**10, 8*1024, 64*1024, 256*1024,
    2*1024*1024, 6*1024*1024, 16*1024*1024, 64*1024*1024,
]
A_SCALAR = 2.0

RUNTIME_RE = re.compile(r"runtime_ms=([\d.]+)")

def compile_saxpy(log):
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

def run(reps=3, warmup=True, randomize=True):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = Path("logs"); results_dir = Path("results")/f"cache_miss_impact_{ts}"; figures_dir = Path("figures")/f"cache_miss_impact_{ts}"
    for d in (logs_dir, results_dir, figures_dir): d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"cache_miss_impact_{ts}.log"
    raw_csv = results_dir / "raw_data.csv"
    summary_csv = results_dir / "summary.csv"

    rt_plot = figures_dir / "runtime_vs_size.png"
    mr_plot = figures_dir / "missrate_vs_size.png"
    corr_plot = figures_dir / "runtime_vs_missrate.png"

    print(f"[INFO] Cache-Miss Impact (sequential)")

    raw_data = {N: [] for N in SIZES}

    with open(log_file,"w") as log, open(raw_csv,"w",newline="") as raw_out:
        writer = csv.writer(raw_out)
        writer.writerow(["rep","N","runtime_ms","cache_refs","cache_misses","miss_rate"])

        utils.log_environment(log, "perf")

        # Compile binary
        compile_saxpy(log)

        # Warmup (largest N, discard)
        if warmup:
            N = max(SIZES)
            cmd = ["perf","stat","-e","cache-references,cache-misses","taskset","-c","0",str(BIN),str(N),str(A_SCALAR)]
            log.write("[WARMUP] " + " ".join(cmd) + "\n")
            subprocess.run(cmd,capture_output=True,text=True)

        # Main reps
        for rep in range(1,reps+1):
            sizes = list(SIZES)
            if randomize: random.shuffle(sizes)
            for N in sizes:
                print(f"[INFO] rep={rep}, N={N}")
                cmd = ["perf","stat","-e","cache-references,cache-misses","taskset","-c","0",str(BIN),str(N),str(A_SCALAR)]
                log.write(f"[RUN] rep={rep}, N={N}\nCommand: " + " ".join(cmd) + "\n")
                res = subprocess.run(cmd,capture_output=True,text=True)
                out, err = res.stdout, res.stderr
                if out: log.write(out+"\n")
                if err: log.write("[STDERR]\n"+err+"\n")

                m = RUNTIME_RE.search(out)
                runtime = float(m.group(1)) if m else None
                refs, misses = parse_perf_output(err)
                miss_rate = (misses/refs) if (refs and misses is not None) else None

                if runtime is not None and miss_rate is not None:
                    raw_data[N].append((runtime,refs,misses,miss_rate))
                    writer.writerow([rep,N,f"{runtime:.4f}",refs,misses,f"{miss_rate:.6f}"])

    # Summaries
    summary_rows = []
    for N,vals in raw_data.items():
        if not vals: continue
        runtimes=[v[0] for v in vals]; missrates=[v[3] for v in vals]
        rt_mean=statistics.mean(runtimes); rt_std=statistics.stdev(runtimes) if len(runtimes)>1 else 0.0
        mr_mean=statistics.mean(missrates); mr_std=statistics.stdev(missrates) if len(missrates)>1 else 0.0
        summary_rows.append((N,rt_mean,rt_std,mr_mean,mr_std))

    with open(summary_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["N","runtime_mean_ms","runtime_std_ms","missrate_mean","missrate_std"])
        for row in summary_rows:
            w.writerow([row[0],f"{row[1]:.4f}",f"{row[2]:.4f}",f"{row[3]:.6f}",f"{row[4]:.6f}"])

    # Plots
    plt.figure(figsize=(8,6))
    summary_rows.sort(key=lambda r:r[0])
    Ns=[r[0] for r in summary_rows]; rt=[r[1] for r in summary_rows]; rt_err=[r[2] for r in summary_rows]
    plt.errorbar(Ns,rt,yerr=rt_err,fmt="o-",capsize=5)
    plt.xscale("log",base=2); plt.xlabel("Problem size N (elements)"); plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Problem Size"); plt.grid(True,linestyle="--",alpha=0.7)
    plt.savefig(rt_plot,dpi=300,bbox_inches="tight"); plt.close()

    plt.figure(figsize=(8,6))
    Ns=[r[0] for r in summary_rows]; mr=[r[3] for r in summary_rows]; mr_err=[r[4] for r in summary_rows]
    plt.errorbar(Ns,mr,yerr=mr_err,fmt="o-",capsize=5)
    plt.xscale("log",base=2); plt.xlabel("Problem size N (elements)"); plt.ylabel("Miss rate")
    plt.title("Miss Rate vs Problem Size"); plt.grid(True,linestyle="--",alpha=0.7)
    plt.savefig(mr_plot,dpi=300,bbox_inches="tight"); plt.close()

    plt.figure(figsize=(8,6))
    mr=[r[3] for r in summary_rows]; rt=[r[1] for r in summary_rows]
    plt.scatter(mr,rt)
    for (x,y,N) in zip(mr,rt,[r[0] for r in summary_rows]):
        plt.annotate(f"N={N}",(x,y),textcoords="offset points",xytext=(5,5),fontsize=8)
    plt.xlabel("Miss Rate"); plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Miss Rate"); plt.grid(True,linestyle="--",alpha=0.7)
    plt.savefig(corr_plot,dpi=300,bbox_inches="tight"); plt.close()

    print(f"[DONE] Results:\n - {raw_csv}\n - {summary_csv}\n - {rt_plot}\n - {mr_plot}\n - {corr_plot}\n - {log_file}")
