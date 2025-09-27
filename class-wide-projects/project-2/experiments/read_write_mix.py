"""
Experiment 3: Read/Write Mix Sweep (Intel MLC v3.11b)
- Four ratios: 100%R, 100%W, 70/30, 50/50
- Uses: mlc --bandwidth_matrix ... -Wn   (n = % writes; default 0 = 100% reads)
- Core pinned with taskset
"""

from pathlib import Path
import subprocess
import csv
import time
import random
import statistics
import re
import matplotlib.pyplot as plt
from experiments import utils

# Path to Intel MLC binary (same convention as other experiments)
MLC_PATH = str(Path.home() / "mlc")

# Force DRAM region (consistent with your other scripts)
BUFFER_SIZE = "256M"

# You can tweak this if you want to explore a different line/stride length.
# Not strictly required by the R/W mix experiment, but harmless to set explicitly.
LINE_LEN = 64  # bytes -> passed via `-l64`

# Ratios: label -> list of flags to append to --bandwidth_matrix
# NOTE: In MLC v3.11b, -Wn controls percent of stores (writes).
#       Default (no -W) is 0% stores -> 100% reads.
RW_CONFIGS = {
    "100R":   [],         # 100% reads (no -W)
    "100W":   ["-W6"],    # 100% writes (Non Temporal Writes, since 100% normal writes is impossible with mlc)
    "70R30W": ["-W9"],    # 30% writes / 70% reads (technically a 3:1 ratio, since it is not possible to get 70/30 with mlc)
    "50R50W": ["-W8"],    # 50% writes / 50% reads
}

def _parse_bandwidth_any(text: str) -> float | None:
    """
    Robust bandwidth parser for MLC --bandwidth_matrix output.
    Strategy:
      1) Matrix rows that start with a node index (e.g., '   0  39926.4  ...')
      2) Lines like 'Memory node 0 bandwidths:  39926.4'
      3) As a fallback, collect any floats on lines that look like matrix rows
    Returns the MAX MB/s observed (consistent with your pattern_granularity behavior).
    """
    lines = utils.filter_and_log(text)
    bw_vals: list[float] = []

    # (1) Matrix rows: first token a node id, subsequent tokens floats
    for line in lines:
        tokens = line.strip().split()
        if tokens and tokens[0].isdigit() and len(tokens) >= 2:
            for tok in tokens[1:]:
                try:
                    bw_vals.append(float(tok))
                except ValueError:
                    pass

    # (2) "bandwidths:" summary form
    if not bw_vals:
        for line in lines:
            if "bandwidths:" in line.lower():
                # Collect all floats on the line
                for tok in re.findall(r"[-+]?\d*\.\d+|\d+", line):
                    try:
                        bw_vals.append(float(tok))
                    except ValueError:
                        pass

    return max(bw_vals) if bw_vals else None

def _parse_loaded_latency_any(text: str) -> float | None:
    """
    Parse latency (ns) from --loaded_latency table.
    Here we return the latency from the first data row (inject=0).
    """
    lines = utils.filter_and_log(text)
    for line in lines:
        tokens = line.strip().split()
        if tokens and tokens[0].isdigit() and len(tokens) >= 3:
            if tokens[0] == "00000":  # inject=0 row
                try:
                    return float(tokens[1])
                except ValueError:
                    return None
    return None


def run(reps: int = 3, warmup: bool = True, randomize: bool = True):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")

    logs_dir = Path("logs")
    results_dir = Path("results") / f"rw_mix_{ts}"
    figures_dir = Path("figures") / f"rw_mix_{ts}"
    for d in (logs_dir, results_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"rw_mix_{ts}.log"
    raw_csv = results_dir / "raw_data.csv"
    summary_csv = results_dir / "summary.csv"
    plot_file = figures_dir / "bandwidth.png"
    md_table = figures_dir / "summary_table.md"

    # ---- NEW: latency outputs (kept separate to avoid changing your existing CSVs) ----
    lat_plot_file = figures_dir / "latency.png"
    summary_lat_csv = results_dir / "summary_latency.csv"
    md_table_lat = figures_dir / "latency_table.md"

    print(f"[INFO] Read/Write Mix Sweep (reps={reps}, warmup={warmup}, randomize={randomize})")

    # ratio -> list of bandwidth samples (unchanged)
    raw_data: dict[str, list[float]] = {label: [] for label in RW_CONFIGS}
    # ---- NEW: ratio -> list of latency samples ----
    raw_lat: dict[str, list[float]] = {label: [] for label in RW_CONFIGS}

    with open(log_file, "w") as log, open(raw_csv, "w", newline="") as raw_out:
        writer = csv.writer(raw_out)
        writer.writerow(["rep", "ratio", "bandwidth_MBps"])

        # Environment / suppression policy (shared)
        utils.log_environment(log, MLC_PATH)

        # -------- Warm-up (discarded) --------
        if warmup:
            log.write("=== Warm-up Phase (discarded from results) ===\n")
            for label, flags in RW_CONFIGS.items():
                cmd = ["taskset", "-c", "0", MLC_PATH, "--bandwidth_matrix",
                       f"-b{BUFFER_SIZE}", f"-l{LINE_LEN}"] + flags
                log.write(f"[WARMUP] {' '.join(cmd)}\n")
                subprocess.run(cmd, capture_output=True, text=True)
            log.write("=== End Warm-up ===\n\n")

        # -------- Main repetitions --------
        configs = list(RW_CONFIGS.items())
        for rep in range(1, reps + 1):
            tests = configs[:]
            if randomize:
                random.shuffle(tests)

            for label, flags in tests:
                print(f"[INFO] rep {rep}/{reps}  ratio={label}")

                # ---- Bandwidth (unchanged) ----
                cmd = ["taskset", "-c", "0", MLC_PATH, "--bandwidth_matrix",
                       f"-b{BUFFER_SIZE}", f"-l{LINE_LEN}"] + flags

                log.write(f"[RUN-BW] rep={rep}, ratio={label}\n")
                log.write("Command: " + " ".join(cmd) + "\n")

                res = subprocess.run(cmd, capture_output=True, text=True)

                # Log both stdout and stderr (some MLC builds print to either)
                for line in utils.filter_and_log(res.stdout):
                    log.write(line + "\n")
                if res.stderr:
                    log.write("[STDERR]\n")
                    for line in utils.filter_and_log(res.stderr):
                        log.write(line + "\n")

                # Combine for parsing robustness
                combined = (res.stdout or "") + "\n" + (res.stderr or "")
                bw = _parse_bandwidth_any(combined)

                if bw is None:
                    log.write("[WARN] No parseable bandwidth found for this run.\n")
                else:
                    raw_data[label].append(bw)
                    writer.writerow([rep, label, f"{bw:.2f}"])

                # ---- NEW: Loaded Latency for the same ratio ----
                cmd_lat = ["taskset", "-c", "0", MLC_PATH, "--loaded_latency",
                           f"-b{BUFFER_SIZE}", f"-l{LINE_LEN}"] + flags
                log.write(f"[RUN-LAT] rep={rep}, ratio={label}\n")
                log.write("Command: " + " ".join(cmd_lat) + "\n")
                res_lat = subprocess.run(cmd_lat, capture_output=True, text=True)

                for line in utils.filter_and_log(res_lat.stdout):
                    log.write(line + "\n")
                if res_lat.stderr:
                    log.write("[STDERR]\n")
                    for line in utils.filter_and_log(res_lat.stderr):
                        log.write(line + "\n")

                lat_ns = _parse_loaded_latency_any((res_lat.stdout or "") + "\n" + (res_lat.stderr or ""))
                if lat_ns is None:
                    log.write("[WARN] No parseable loaded-latency found for this run.\n")
                else:
                    raw_lat[label].append(lat_ns)

    # -------- Summaries (bandwidth - unchanged) --------
    summary_rows = []
    for label, vals in raw_data.items():
        if vals:
            mean_bw = statistics.mean(vals)
            std_bw = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary_rows.append((label, mean_bw, std_bw))

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ratio", "bw_mean_MBps", "bw_std_MBps"])
        for row in summary_rows:
            w.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.2f}"])

    # Also a small Markdown table for report copy/paste (bandwidth)
    with open(md_table, "w") as f:
        f.write("| Ratio | Mean BW (MB/s) | Std (MB/s) |\n")
        f.write("|-------|-----------------|------------|\n")
        for label, mean_bw, std_bw in summary_rows:
            f.write(f"| {label} | {mean_bw:.2f} | {std_bw:.2f} |\n")

        # -------- Plot --------
    # Custom order: descending reads (100R -> 70R30W -> 50R50W -> 100W)
    desired_order = ["100R", "70R30W", "50R50W", "100W"]
    order_map = {label: i for i, label in enumerate(desired_order)}

    # Filter and sort rows according to desired order
    sorted_rows = sorted(summary_rows, key=lambda r: order_map.get(r[0], 999))

    labels = [r[0] for r in sorted_rows]
    means  = [r[1] for r in sorted_rows]
    stds   = [r[2] for r in sorted_rows]

    plt.figure(figsize=(8, 6))
    plt.errorbar(labels, means, yerr=stds, marker="o", capsize=6)
    plt.ylabel("Bandwidth (MB/s)")
    plt.xlabel("Read/Write Ratio")
    plt.title("Read/Write Mix Sweep (MLC --bandwidth_matrix, DRAM)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    # -------- NEW: Latency summaries/plot (kept separate) --------
    lat_summary_rows = []
    for label, vals in raw_lat.items():
        if vals:
            mean_lat = statistics.mean(vals)
            std_lat = statistics.stdev(vals) if len(vals) > 1 else 0.0
            lat_summary_rows.append((label, mean_lat, std_lat))

    # Save latency summary CSV (separate file to avoid changing your existing CSV schema)
    with open(summary_lat_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ratio", "lat_mean_ns", "lat_std_ns"])
        for row in lat_summary_rows:
            w.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.2f}"])

    # Latency markdown table
    with open(md_table_lat, "w") as f:
        f.write("| Ratio | Mean Latency (ns) | Std (ns) |\n")
        f.write("|-------|--------------------|----------|\n")
        for label, mean_lat, std_lat in lat_summary_rows:
            f.write(f"| {label} | {mean_lat:.2f} | {std_lat:.2f} |\n")

    # Plot latency (same x-order)
    lat_sorted = sorted(lat_summary_rows, key=lambda r: order_map.get(r[0], 999))
    lat_labels = [r[0] for r in lat_sorted]
    lat_means  = [r[1] for r in lat_sorted]
    lat_stds   = [r[2] for r in lat_sorted]

    plt.figure(figsize=(8, 6))
    plt.errorbar(lat_labels, lat_means, yerr=lat_stds, marker="o", capsize=6)
    plt.ylabel("Latency (ns)")
    plt.xlabel("Read/Write Ratio")
    plt.title("Read/Write Mix Sweep (MLC --loaded_latency, DRAM)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(lat_plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Results:\n - {raw_csv}\n - {summary_csv}\n - {md_table}\n - {plot_file}\n"
          f" - {summary_lat_csv}\n - {md_table_lat}\n - {lat_plot_file}\n - {log_file}")
