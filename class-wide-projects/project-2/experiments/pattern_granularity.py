"""
Experiment 2: Pattern & Granularity Sweep (Intel MLC v3.11b)
- Measures latency with --idle_latency
- Measures bandwidth with --bandwidth_matrix
- Patterns: sequential vs random
- Strides: ≈64B, ≈256B, ≈1024B
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

MLC_PATH = str(Path.home() / "mlc")

# Patterns: flags for idle_latency and bandwidth_matrix
PATTERNS = {
    "sequential": {"lat": [], "bw": []},
    "random": {"lat": ["-r"], "bw": ["-R"]},
}

# Strides in bytes
STRIDES = {
    "64B": 64,
    "256B": 256,
    "1024B": 1024,
}

BUFFER_SIZE = "256M"   # Force DRAM
CPU_FREQ_GHZ = 2.5     # Used for cycle conversion if needed


# ---------------- Parsing helpers ----------------
def parse_latency(output: str):
    for line in utils.filter_and_log(output):
        if "iteration took" in line:
            m = re.compile(r"\(\s*([\d.]+)\s*ns\)").search(line)
            if m:
                return float(m.group(1))
    return None

def parse_bandwidth(output: str):
    """
    Parse bandwidth (MB/s) from MLC --bandwidth_matrix output.
    Looks for matrix rows: lines that start with a NUMA node index.
    """
    bw_vals = []
    for line in utils.filter_and_log(output):
        # Example row: "       0    39926.4"
        tokens = line.strip().split()
        if not tokens:
            continue
        # Row should start with an integer node id
        if tokens[0].isdigit() and len(tokens) >= 2:
            try:
                # Collect all numeric values after the node index
                for tok in tokens[1:]:
                    bw_vals.append(float(tok))
            except ValueError:
                continue
    return max(bw_vals) if bw_vals else None


# ---------------- Main runner ----------------
def run(reps=3, warmup=True, randomize=True):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")

    logs_dir = Path("logs")
    results_dir = Path("results") / f"pattern_granularity_{ts}"
    figures_dir = Path("figures") / f"pattern_granularity_{ts}"
    for d in (logs_dir, results_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"pattern_granularity_{ts}.log"
    raw_csv = results_dir / "raw_data.csv"
    summary_csv = results_dir / "summary.csv"
    latency_plot = figures_dir / "latency.png"
    bw_plot = figures_dir / "bandwidth.png"

    print(f"[INFO] Pattern & Granularity Sweep (reps={reps}, warmup={warmup}, randomize={randomize})")

    raw_data = {(p, s): [] for p in PATTERNS for s in STRIDES}

    with open(log_file, "w") as log, open(raw_csv, "w", newline="") as raw_out:
        writer = csv.writer(raw_out)
        writer.writerow(["rep", "pattern", "stride", "latency_ns", "bandwidth_MBps"])

        utils.log_environment(log, MLC_PATH)

        # Warmup phase
        if warmup:
            log.write("=== Warm-up Phase (discarded) ===\n")
            for pname, flags in PATTERNS.items():
                for ssize in STRIDES.values():
                    # idle_latency
                    cmd_lat = ["taskset", "-c", "0", MLC_PATH, "--idle_latency",
                               f"-b{BUFFER_SIZE}", f"-l{ssize}"] + flags["lat"]
                    log.write(f"[WARMUP] {' '.join(cmd_lat)}\n")
                    subprocess.run(cmd_lat, capture_output=True, text=True)
                    # bandwidth_matrix
                    cmd_bw = ["taskset", "-c", "0", MLC_PATH, "--bandwidth_matrix",
                              f"-b{BUFFER_SIZE}", f"-l{ssize}"] + flags["bw"]
                    log.write(f"[WARMUP] {' '.join(cmd_bw)}\n")
                    subprocess.run(cmd_bw, capture_output=True, text=True)
            log.write("=== End Warm-up ===\n\n")

        combos = list(PATTERNS.items())
        stride_items = list(STRIDES.items())

        for rep in range(1, reps + 1):
            tests = [(pname, flags, sname, ssize) for pname, flags in combos for sname, ssize in stride_items]
            if randomize:
                random.shuffle(tests)

            for pname, flags, sname, ssize in tests:
                print(f"[INFO] rep {rep}/{reps}  pattern={pname:9s}  stride={sname}")

                # --- Latency ---
                cmd_lat = ["taskset", "-c", "0", MLC_PATH, "--idle_latency",
                           f"-b{BUFFER_SIZE}", f"-l{ssize}"] + flags["lat"]
                log.write(f"[RUN-LAT] rep={rep}, pattern={pname}, stride={sname}\n")
                log.write("Command: " + " ".join(cmd_lat) + "\n")
                res_lat = subprocess.run(cmd_lat, capture_output=True, text=True)
                for line in utils.filter_and_log(res_lat.stdout):
                    log.write(line + "\n")
                lat = parse_latency(res_lat.stdout)

                # --- Bandwidth ---
                cmd_bw = ["taskset", "-c", "0", MLC_PATH, "--bandwidth_matrix",
                          f"-b{BUFFER_SIZE}", f"-l{ssize}"] + flags["bw"]
                log.write(f"[RUN-BW] rep={rep}, pattern={pname}, stride={sname}\n")
                log.write("Command: " + " ".join(cmd_bw) + "\n")
                res_bw = subprocess.run(cmd_bw, capture_output=True, text=True)
                for line in utils.filter_and_log(res_bw.stdout):
                    log.write(line + "\n")
                bw = parse_bandwidth(res_bw.stdout)

                if lat is not None and bw is not None:
                    raw_data[(pname, sname)].append((lat, bw))
                    writer.writerow([rep, pname, sname, f"{lat:.2f}", f"{bw:.2f}"])

    # Summaries
    summary_rows = []
    for (pname, sname), vals in raw_data.items():
        if vals:
            lats = [v[0] for v in vals]
            bws = [v[1] for v in vals]
            summary_rows.append((
                pname, sname,
                statistics.mean(lats), statistics.stdev(lats) if len(lats) > 1 else 0.0,
                statistics.mean(bws), statistics.stdev(bws) if len(bws) > 1 else 0.0
            ))

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "stride", "lat_mean_ns", "lat_std_ns",
                    "bw_mean_MBps", "bw_std_MBps"])
        for row in summary_rows:
            w.writerow(row)

    # Plots
        # -------- Relative Difference Plots (Sequential vs Random) --------
    strides = [s for _, s, *_ in summary_rows if _ == "sequential"]
    rel_bw = []
    rel_lat = []

    for sname in STRIDES.keys():
        # Grab matching entries
        seq_row = next((r for r in summary_rows if r[0] == "sequential" and r[1] == sname), None)
        rand_row = next((r for r in summary_rows if r[0] == "random" and r[1] == sname), None)
        if seq_row and rand_row:
            bw_ratio = seq_row[4] / rand_row[4] if rand_row[4] > 0 else float("nan")
            lat_ratio = rand_row[2] / seq_row[2] if seq_row[2] > 0 else float("nan")
            rel_bw.append((sname, bw_ratio))
            rel_lat.append((sname, lat_ratio))

    # Bandwidth speedup
    if rel_bw:
        plt.figure(figsize=(8,6))
        xs = [x for x, _ in rel_bw]
        ys = [y for _, y in rel_bw]
        plt.bar(xs, ys, color="skyblue")
        plt.axhline(1.0, color="k", linestyle="--")
        plt.ylabel("Sequential / Random Bandwidth")
        plt.title("Relative Bandwidth: Sequential vs Random")
        plt.savefig(figures_dir / "relative_bandwidth.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {figures_dir / 'relative_bandwidth.png'}")

    # Latency ratio
    if rel_lat:
        plt.figure(figsize=(8,6))
        xs = [x for x, _ in rel_lat]
        ys = [y for _, y in rel_lat]
        plt.bar(xs, ys, color="salmon")
        plt.axhline(1.0, color="k", linestyle="--")
        plt.ylabel("Random / Sequential Latency")
        plt.title("Relative Latency: Sequential vs Random")
        plt.savefig(figures_dir / "relative_latency.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {figures_dir / 'relative_latency.png'}")

    for metric, ylabel, idx_mean, idx_std, out_path in [
        ("latency", "Latency (ns)", 2, 3, latency_plot),
        ("bandwidth", "Bandwidth (MB/s)", 4, 5, bw_plot),
    ]:
        plt.figure(figsize=(8, 6))
        for pname in PATTERNS:
            xs = [r[1] for r in summary_rows if r[0] == pname]
            ys = [r[idx_mean] for r in summary_rows if r[0] == pname]
            es = [r[idx_std] for r in summary_rows if r[0] == pname]
            if xs:
                plt.errorbar(xs, ys, yerr=es, marker="o", capsize=6, label=pname)
        plt.xlabel("Stride")
        plt.ylabel(ylabel)
        plt.title(f"Pattern × Granularity Sweep: {metric}")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {out_path}")

    print(f"[DONE] Results:\n - {raw_csv}\n - {summary_csv}\n - {latency_plot}\n - {bw_plot}\n - {log_file}")
