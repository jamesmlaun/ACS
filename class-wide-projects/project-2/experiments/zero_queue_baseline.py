"""
Zero-queue baseline measurement using Intel MLC.
Measures L1, L2, L3, and DRAM latencies using --idle_latency with buffer sizes.
- Core pinned with taskset
- Warm-up runs (discarded from results, optional)
- Randomized buffer order (optional)
- Universal environment logging handled via utils.log_environment()
"""

import subprocess
import csv
import time
import random
from pathlib import Path
import matplotlib.pyplot as plt
import statistics
import re
from experiments import utils

# Path to Intel MLC binary
MLC_PATH = str(Path.home() / "mlc")

# Buffer sizes chosen to isolate each cache level (Intel recommended practice)
BUFFER_SIZES = {
    "L1": "20K",
    "L2": "640K",
    "L3": "8M",
    "DRAM": "256M"
}

# CPU base frequency in GHz (adjust if needed)
CPU_FREQ_GHZ = 2.5


def run(reps=3, warmup=True, randomize=True):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Output directories
    logs_dir = Path("logs")
    results_dir = Path("results") / f"zero_queue_baseline_{ts}"
    figures_dir = Path("figures") / f"zero_queue_baseline_{ts}"
    for d in [logs_dir, results_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # File paths
    log_file = logs_dir / f"zero_queue_baseline_{ts}.log"
    raw_csv = results_dir / "raw_data.csv"
    summary_csv = results_dir / "summary.csv"
    plot_file = figures_dir / "latency.png"
    md_table = figures_dir / "summary_table.md"

    print(f"[INFO] Starting zero-queue baseline ({reps} reps, warmup={warmup}, randomize={randomize})...")

    raw_data = {lvl: [] for lvl in BUFFER_SIZES.keys()}

    with open(log_file, "w") as log, open(raw_csv, "w", newline="") as raw_out:
        writer = csv.writer(raw_out)
        writer.writerow(["repetition", "level", "latency_ns"])

        # --- Universal environment logging (suppression + CPU info + tool versions) ---
        utils.log_environment(log, MLC_PATH)

        # --- Warm-up phase ---
        if warmup:
            log.write("=== Warm-up Phase (discarded from results) ===\n")
            for level, buf_size in BUFFER_SIZES.items():
                cmd = ["taskset", "-c", "0", MLC_PATH, "--idle_latency", f"-b{buf_size}"]
                log.write(f"[WARMUP] Running: {' '.join(cmd)}\n")
                result = subprocess.run(cmd, capture_output=True, text=True)
                for line in utils.filter_and_log(result.stdout):
                    log.write(line + "\n")
            log.write("=== End Warm-up ===\n\n")

        # --- Main repetitions ---
        for i in range(reps):
            levels = list(BUFFER_SIZES.items())
            if randomize:
                random.shuffle(levels)

            for level, buf_size in levels:
                print(f"[INFO] Repetition {i+1}/{reps}, level {level} (buffer {buf_size})...")
                cmd = ["taskset", "-c", "0", MLC_PATH, "--idle_latency", f"-b{buf_size}"]
                log.write(f"[RUN] Repetition {i+1}, {level} ({buf_size})\n")
                log.write(f"Command: {' '.join(cmd)}\n")

                result = subprocess.run(cmd, capture_output=True, text=True)

                for line in utils.filter_and_log(result.stdout):
                    log.write(line + "\n")

                if result.stderr:
                    for line in utils.filter_and_log(result.stderr):
                        log.write("STDERR:\n" + line + "\n")

                for line in result.stdout.splitlines():
                    if "iteration took" in line:
                        match = re.search(r"\(\s*([\d.]+)\s*ns\)", line)
                        if match:
                            latency = float(match.group(1))
                            raw_data[level].append(latency)
                            writer.writerow([i + 1, level, latency])

    print(f"[INFO] Execution log written to {log_file}")
    print(f"[INFO] Raw results saved to {raw_csv}")

    # --- Compute summary stats ---
    summary_rows = []
    for level, values in raw_data.items():
        if values:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            cycles = mean_val * CPU_FREQ_GHZ  # ns * GHz = cycles
            summary_rows.append((level, mean_val, std_val, cycles))

    # --- Save summary CSV ---
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "mean_ns", "stddev_ns", "mean_cycles"])
        for level, mean_val, std_val, cycles in summary_rows:
            writer.writerow([level, f"{mean_val:.2f}", f"{std_val:.2f}", f"{cycles:.2f}"])
        writer.writerow([])
        writer.writerow(["NOTE", "Prefetchers could not be disabled under WSL2; MLC used random access mode."])

    print(f"[INFO] Summary saved to {summary_csv}")

    # --- Save Markdown table in figures directory ---
    with open(md_table, "w") as f:
        f.write("| Level | Mean (ns) | Stddev (ns) | Mean (cycles) |\n")
        f.write("|-------|-----------|-------------|----------------|\n")
        for level, mean_val, std_val, cycles in summary_rows:
            f.write(f"| {level} | {mean_val:.2f} | {std_val:.2f} | {cycles:.2f} |\n")
        f.write("\n*Notes:*\n")
        f.write("- Prefetchers could not be disabled under WSL2; MLC used random access mode.\n")

    print(f"[INFO] Markdown table saved to {md_table}")

    # --- Plot results ---
    levels = [row[0] for row in summary_rows]
    means = [row[1] for row in summary_rows]
    stds = [row[2] for row in summary_rows]

    plt.figure(figsize=(8, 6))
    plt.bar(levels, means, yerr=stds, capsize=6, color="skyblue", label="Latency (ns)")
    plt.ylabel("Latency (ns)")
    plt.title("Zero-Queue Baseline Latencies (MLC idle_latency, core pinned)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved to {plot_file}")
    print("[INFO] Experiment completed successfully.")
