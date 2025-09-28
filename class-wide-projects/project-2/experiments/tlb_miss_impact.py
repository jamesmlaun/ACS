#!/usr/bin/env python3
"""
TLB-miss impact experiment using saxpy_tlb.
- Varies number of distinct pages (2^k) and huge-pages ON/OFF
- Executes constant work (passes * pages * lines_per_page updates)
- Collects runtime and DTLB miss counters with perf
- Captures THP diagnostics from kernel stderr (smaps AnonHugePages before/after)
- Outputs raw CSV, summary CSV, and runtime vs pages plot
"""

import subprocess
import csv
import time
import random
import statistics
import re
from pathlib import Path
import matplotlib.pyplot as plt
from experiments import utils

# --- Config ---
CXX = "g++"
SRC = Path("saxpy_tlb.cpp")        # compile from repo root
BIN = Path("build") / "saxpy_tlb"  # output binary

PAGE_BYTES = 4096
LINES_PER_PAGE = 1
LINE_BYTES = 64
TOTAL_UPDATES = 50_000_000
ALPHA = 2.0
PATTERN = "random"
HUGE_MODES = ["off", "on"]

PAGES_POW = list(range(6, 18))  # 2^6=64 → 2^17=131072 pages

PERF_EVENTS = [
    "cycles",
    "instructions",
    "task-clock",
    "dTLB-load-misses",
    "dTLB-store-misses",
    "dtlb_load_misses.walk_completed",
    "dtlb_store_misses.walk_completed",
]

# stdout from kernel
RUNTIME_RE = re.compile(r"runtime_ms=([0-9.]+)")
GFLOPS_RE  = re.compile(r"gflops=([0-9.]+)")

# stderr from kernel (THP diagnostics)
THP_SMAPS_BEFORE_RE = re.compile(r"\[THP\]\s*/proc/self/smaps AnonHugePages total:\s*([0-9]+)\s*kB\s*\(before\)")
THP_SMAPS_AFTER_RE  = re.compile(r"\[THP\]\s*/proc/self/smaps AnonHugePages total:\s*([0-9]+)\s*kB\s*\(after\)")

def compile_kernel(log):
    BIN.parent.mkdir(parents=True, exist_ok=True)
    cmd = [CXX, "-O3", "-march=native", "-std=c++17", str(SRC), "-o", str(BIN)]
    log.write("[COMPILE] " + " ".join(cmd) + "\n")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout: log.write(res.stdout + "\n")
    if res.stderr: log.write(res.stderr + "\n")
    if res.returncode != 0:
        raise RuntimeError("Compilation failed.")

def parse_perf_err(text: str):
    vals = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            val, _, evt = parts[:3]
            try:
                vals[evt] = float(val.replace(",", ""))
            except:
                pass
    return vals

def compute_passes(pages: int) -> tuple[int, int]:
    updates_per_pass = max(1, pages * LINES_PER_PAGE)
    passes = max(1, TOTAL_UPDATES // updates_per_pass)
    updates = passes * updates_per_pass
    return passes, updates

def run(reps=3, warmup=True, randomize=True):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")

    logs_dir = Path("logs")
    results_dir = Path("results") / f"tlb_miss_impact_{ts}"
    figures_dir = Path("figures") / f"tlb_miss_impact_{ts}"
    for d in (logs_dir, results_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_file    = logs_dir / f"tlb_miss_impact_{ts}.log"
    raw_csv     = results_dir / "raw_data.csv"
    summary_csv = results_dir / "summary.csv"
    plot_file   = figures_dir / "runtime_vs_pages.png"

    print(f"[INFO] Starting TLB-miss impact ({reps} reps, warmup={warmup}, randomize={randomize})...")

    raw_data = {(h, 1 << k): [] for h in HUGE_MODES for k in PAGES_POW}

    with open(log_file, "w") as log, open(raw_csv, "w", newline="") as raw_out:
        writer = csv.writer(raw_out)
        writer.writerow([
            "rep", "huge", "pattern", "pages", "passes", "updates",
            "runtime_ms", "gflops",
            "dtlb_load_misses", "dtlb_store_misses",
            "dtlb_walks_load", "dtlb_walks_store", "misses_per_ku",
            "smaps_kb_before", "smaps_kb_after", "thp_effective"
        ])

        # Environment info
        utils.log_environment(log, "perf")
        for path in ("/proc/meminfo",
                     "/sys/kernel/mm/transparent_hugepage/enabled",
                     "/sys/kernel/mm/transparent_hugepage/defrag"):
            try:
                txt = Path(path).read_text()
                log.write(f"[ENV] {path}\n{txt}\n")
            except Exception as e:
                log.write(f"[WARN] Could not read {path}: {e}\n")

        # Build kernel
        compile_kernel(log)

        # --- Warm-up (discarded) ---
        if warmup:
            pages = 1 << PAGES_POW[0]
            passes, _ = compute_passes(pages)
            args = [str(pages), str(PAGE_BYTES), str(LINES_PER_PAGE), str(LINE_BYTES),
                    str(passes), str(ALPHA), PATTERN, "off"]
            cmd = ["taskset", "-c", "0", str(BIN)] + args
            log.write("[WARMUP] " + " ".join(cmd) + "\n")
            subprocess.run(cmd, capture_output=True, text=True)

        # --- Main repetitions ---
        for rep in range(1, reps + 1):
            configs = [(h, 1 << k) for h in HUGE_MODES for k in PAGES_POW]
            if randomize:
                random.shuffle(configs)

            for huge, pages in configs:
                passes, updates = compute_passes(pages)
                print(f"[INFO] rep={rep}, huge={huge}, pages={pages}, passes={passes}")

                args = [str(pages), str(PAGE_BYTES), str(LINES_PER_PAGE), str(LINE_BYTES),
                        str(passes), str(ALPHA), PATTERN, huge]
                cmd = ["perf", "stat", "-x", ",", "-e", ",".join(PERF_EVENTS),
                       "taskset", "-c", "0", str(BIN)] + args

                log.write(f"[RUN] rep={rep}, huge={huge}, pages={pages}\n")
                log.write("Command: " + " ".join(cmd) + "\n")

                res = subprocess.run(cmd, capture_output=True, text=True)
                out, err = res.stdout, res.stderr

                if out: log.write(out + "\n")
                if err: log.write("[STDERR]\n" + err + "\n")

                # Parse kernel stdout
                runtime, gflops = None, None
                for line in (out or "").splitlines():
                    m = RUNTIME_RE.search(line)
                    if m: runtime = float(m.group(1))
                    m = GFLOPS_RE.search(line)
                    if m: gflops = float(m.group(1))
                if runtime is not None and gflops is None:
                    gflops = (updates * 2.0) / (runtime * 1e6)

                # Parse perf stderr
                perf_vals = parse_perf_err(err or "")
                lmiss = perf_vals.get("dTLB-load-misses")
                smiss = perf_vals.get("dTLB-store-misses")
                wld   = perf_vals.get("dtlb_load_misses.walk_completed")
                wst   = perf_vals.get("dtlb_store_misses.walk_completed")
                total_misses = sum(v for v in (lmiss, smiss, wld, wst) if v is not None)
                mpku = (total_misses / max(1, updates)) * 1000.0 if runtime is not None else float("nan")

                # Parse THP diagnostics from stderr
                smaps_before = None
                smaps_after  = None
                for line in (err or "").splitlines():
                    m1 = THP_SMAPS_BEFORE_RE.search(line)
                    if m1: smaps_before = int(m1.group(1))
                    m2 = THP_SMAPS_AFTER_RE.search(line)
                    if m2: smaps_after = int(m2.group(1))
                thp_effective = (smaps_before is not None and smaps_after is not None and smaps_after > smaps_before)

                if runtime is not None:
                    raw_data[(huge, pages)].append((runtime, gflops, lmiss, smiss, wld, wst, mpku,
                                                    smaps_before, smaps_after, thp_effective))
                    writer.writerow([rep, huge, PATTERN, pages, passes, updates,
                                     f"{runtime:.4f}", f"{gflops:.4f}",
                                     lmiss or "", smiss or "", wld or "", wst or "",
                                     f"{mpku:.6f}",
                                     smaps_before if smaps_before is not None else "",
                                     smaps_after if smaps_after is not None else "",
                                     int(thp_effective)])

    # --- Summaries ---
    summary_rows = []
    for (huge, pages), vals in raw_data.items():
        if not vals:
            continue
        rts   = [v[0] for v in vals]
        gfs   = [v[1] for v in vals]
        mpkus = [v[6] for v in vals]
        thp_ok = [1 if v[9] else 0 for v in vals]  # thp_effective
        rt_mean = statistics.mean(rts)
        rt_std  = statistics.stdev(rts) if len(rts) > 1 else 0.0
        gf_mean = statistics.mean(gfs)
        gf_std  = statistics.stdev(gfs) if len(gfs) > 1 else 0.0
        mr_mean = statistics.mean(mpkus) if len(mpkus) > 0 else float("nan")
        mr_std  = statistics.stdev(mpkus) if len(mpkus) > 1 else 0.0
        thp_rate = sum(thp_ok) / len(thp_ok)
        summary_rows.append((huge, pages, rt_mean, rt_std, gf_mean, gf_std, mr_mean, mr_std, thp_rate))

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["huge","pages",
                    "runtime_mean_ms","runtime_std_ms",
                    "gflops_mean","gflops_std",
                    "mpku_mean","mpku_std",
                    "thp_effective_rate"])
        for row in sorted(summary_rows, key=lambda r: (r[0], r[1])):
            w.writerow([row[0], row[1],
                        f"{row[2]:.4f}", f"{row[3]:.4f}",
                        f"{row[4]:.4f}", f"{row[5]:.4f}",
                        f"{row[6]:.6f}", f"{row[7]:.6f}",
                        f"{row[8]:.2f}"])

    # --- Plot: Runtime vs Pages (series by huge-pages) with error bars ---
    plt.figure(figsize=(8, 6))
    for huge in HUGE_MODES:
        pts = [(pages, rt, rt_std) for (h, pages, rt, rt_std, _, _, _, _, _) in summary_rows if h == huge]
        pts.sort(key=lambda t: t[0])
        if pts:
            xs  = [x for x, _, _ in pts]
            ys  = [y for _, y, _ in pts]
            yerr = [e for _, _, e in pts]
            plt.errorbar(xs, ys, yerr=yerr, fmt="o-", capsize=4, label=f"huge={huge}")
    plt.xlabel("Number of distinct pages")
    plt.ylabel("Runtime (ms)")
    plt.title("TLB-Miss Impact (constant work, pages sweep)")
    plt.xscale("log", base=2)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


    print(f"[DONE] Results:\n - {raw_csv}\n - {summary_csv}\n - {plot_file}\n - {log_file}")
