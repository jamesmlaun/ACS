"""
Experiment 5: Working-Set Size Sweep (Intel MLC v3.11b)
- Measures latency with --idle_latency across working-set sizes.
- Shows transitions at known cache boundaries for i7-11850H:
    L1 = 48 KiB, L2 = 1.25 MiB, L3 = 24 MiB.
- Annotates L1/L2/L3/DRAM regions on the plot.

Outputs:
  logs/working_set_sweep_<ts>.log
  results/.../raw_data.csv, summary.csv
  figures/.../latency.png
"""

import subprocess
import csv
import time
import random
import statistics
import re
import math
from pathlib import Path
import matplotlib.pyplot as plt
from experiments import utils

MLC_PATH = str(Path.home() / "mlc")

# ------------------ Hardware boundaries ------------------
L1_BYTES = 48 * 1024
L2_BYTES = int(1.25 * (1 << 20))   # 1.25 MiB
L3_BYTES = 24 * (1 << 20)          # 24 MiB

# ------------------ Sizes to sweep ------------------
# Representative set to show each region
SIZES_BYTES = [
    # L1
    8 * 1024,          # 8 KiB
    32 * 1024,         # 32 KiB
    # L2
    256 * 1024,        # 256 KiB
    1 * 1024 * 1024,   # 1 MiB
    # L3
    4 * 1024 * 1024,   # 4 MiB
    8 * 1024 * 1024,   # 8 MiB
    20 * 1024 * 1024,  # just below 24 MiB
    32 * 1024 * 1024,  # just above 24 MiB
    # DRAM plateau
    64 * 1024 * 1024,   # 64 MiB
    128 * 1024 * 1024,  # 128 MiB
    256 * 1024 * 1024   # 256 MiB
]

# Regex to parse latency lines from MLC
LAT_RE = re.compile(r"\(\s*([\d.]+)\s*ns\)")

# ------------------ Helpers ------------------
def _bytes_to_label(n: int) -> str:
    if n >= (1 << 30): return f"{n/(1<<30):.1f} GiB"
    if n >= (1 << 20): return f"{n/(1<<20):.1f} MiB"
    if n >= (1 << 10): return f"{n/(1<<10):.1f} KiB"
    return f"{n} B"

def _format_size_for_mlc(n_bytes: int) -> str:
    """Format byte size into K/M/G suffix for MLC -b flag (expects units)."""
    if n_bytes % (1 << 30) == 0:
        return f"{n_bytes // (1 << 30)}G"
    elif n_bytes % (1 << 20) == 0:
        return f"{n_bytes // (1 << 20)}M"
    elif n_bytes % (1 << 10) == 0:
        return f"{n_bytes // (1 << 10)}K"
    else:
        # fallback: round to nearest KiB
        return f"{round(n_bytes/(1<<10))}K"

# ------------------ Runner ------------------
def run(reps=3, warmup=True, randomize=True):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")

    logs_dir = Path("logs")
    results_dir = Path("results") / f"working_set_sweep_{ts}"
    figures_dir = Path("figures") / f"working_set_sweep_{ts}"
    for d in (logs_dir, results_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"working_set_sweep_{ts}.log"
    raw_csv = results_dir / "raw_data.csv"
    summary_csv = results_dir / "summary.csv"
    plot_file = figures_dir / "latency.png"

    print(f"[INFO] Working-set sweep (reps={reps}, warmup={warmup}, randomize={randomize})")

    raw_data: dict[int, list[float]] = {s: [] for s in SIZES_BYTES}

    with open(log_file, "w") as log, open(raw_csv, "w", newline="") as raw_out:
        writer = csv.writer(raw_out)
        writer.writerow(["rep", "size_bytes", "size_label", "latency_ns"])

        # Environment info
        utils.log_environment(log, MLC_PATH)

        # Warm-up phase
        if warmup:
            log.write("=== Warm-up Phase (discarded) ===\n")
            for b in (SIZES_BYTES[0], SIZES_BYTES[-1]):
                buf_arg = _format_size_for_mlc(b)
                cmd = ["taskset", "-c", "0", MLC_PATH, "--idle_latency", f"-b{buf_arg}"]
                log.write(f"[WARMUP] {' '.join(cmd)}\n")
                subprocess.run(cmd, capture_output=True, text=True)
            log.write("=== End Warm-up ===\n\n")

        # Main repetitions
        for rep in range(1, reps + 1):
            sizes = list(SIZES_BYTES)
            if randomize:
                random.shuffle(sizes)
            for b in sizes:
                buf_arg = _format_size_for_mlc(b)
                print(f"[INFO] rep {rep}/{reps}, size={_bytes_to_label(b)}")
                cmd = ["taskset", "-c", "0", MLC_PATH, "--idle_latency", f"-b{buf_arg}"]
                log.write(f"[RUN] rep={rep}, size={b} ({_bytes_to_label(b)})\n")
                log.write("Command: " + " ".join(cmd) + "\n")

                res = subprocess.run(cmd, capture_output=True, text=True)

                # Log filtered output
                for line in utils.filter_and_log(res.stdout):
                    log.write(line + "\n")
                if res.stderr:
                    for line in utils.filter_and_log(res.stderr):
                        log.write("STDERR: " + line + "\n")

                # Parse latency
                latency = None
                for line in res.stdout.splitlines():
                    m = LAT_RE.search(line)
                    if m:
                        latency = float(m.group(1))
                        break

                if latency is not None:
                    raw_data[b].append(latency)
                    writer.writerow([rep, b, _bytes_to_label(b), f"{latency:.2f}"])

    # ---- Summaries ----
    summary_rows = []
    for b in SIZES_BYTES:
        vals = raw_data[b]
        if vals:
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary_rows.append((b, _bytes_to_label(b), mu, sd))

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size_bytes", "size_label", "mean_ns", "stddev_ns"])
        for b, lbl, mu, sd in summary_rows:
            w.writerow([b, lbl, f"{mu:.2f}", f"{sd:.2f}"])

    # ---- Plot ----
    plt.figure(figsize=(8, 6))
    x_mib = [b / (1024*1024) for b, _, _, _ in summary_rows]
    y = [mu for _, _, mu, _ in summary_rows]
    yerr = [sd for _, _, _, sd in summary_rows]

    plt.errorbar(x_mib, y, yerr=yerr, fmt="o-", capsize=6, label="Latency (ns)")
    plt.xscale("log", base=2)
    plt.xlabel("Working-set size (MiB, log scale)")
    plt.ylabel("Latency (ns)")
    plt.title("Working-Set Size Sweep")

    ymax = max(y) * 1.1

    # Mark cache boundaries
    boundaries = [
        (L1_BYTES/(1<<20), "L1→L2"),
        (L2_BYTES/(1<<20), "L2→L3"),
        (L3_BYTES/(1<<20), "L3→DRAM"),
    ]
    for xb, lab in boundaries:
        plt.axvline(xb, linestyle=":", color="gray")
        plt.annotate(lab, xy=(xb, ymax*0.9), xytext=(0, 5),
                     textcoords="offset points", ha="center", fontsize=8)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Results:\n - {raw_csv}\n - {summary_csv}\n - {plot_file}\n - {log_file}")
