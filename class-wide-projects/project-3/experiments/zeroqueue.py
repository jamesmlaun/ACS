#!/usr/bin/env python3
"""
Zero-Queue Baseline Experiment (QD=1)

Runs:
- 4 KiB random read
- 4 KiB random write
- 128 KiB sequential read
- 128 KiB sequential write

Features:
- Supports repetitions (--reps)
- Optional randomization of trial order (--randomize)
- Uses incompressible data pattern flags by default
- Logs progress + metrics to unified logfile
- Outputs per-rep CSV, summary CSV, and bar chart with error bars
"""

import subprocess
import json
import csv
import argparse
import statistics
import matplotlib.pyplot as plt
from pathlib import Path
from utils import (
    TARGET_FILE,
    prepare_run_dirs,
    log_msg,
    randomize_workloads,
    fio_cmd,
)

# ----------------------------
# Parameters
# ----------------------------
SIZE = "1G"
RUNTIME = 30
BS_SETTINGS = [
    ("4k", "randread"),
    ("4k", "randwrite"),
    ("128k", "read"),
    ("128k", "write"),
]

# ----------------------------
# Helper Functions
# ----------------------------
def run_fio(bs, rw, out_json, logfile, rep_idx):
    """Run fio for given workload and return parsed JSON."""
    name = f"zeroqueue_{bs}_{rw}_rep{rep_idx}"
    cmd = fio_cmd(
        name=name,
        filename=TARGET_FILE,
        rw=rw,
        bs=bs,
        size=SIZE,
        runtime=RUNTIME,
        iodepth=1,
        numjobs=1,
        out_json=out_json
    )
    log_msg(f"Launching FIO: {name}", logfile)
    subprocess.run(cmd, check=True)
    log_msg(f"Completed FIO: {name}", logfile)

    with open(out_json) as f:
        return json.load(f)

def extract_metrics(data):
    """Extract avg, p95, p99 latency (µs) from fio JSON."""
    job = data["jobs"][0]
    rwtype = "read" if job["read"]["io_bytes"] > 0 else "write"
    lat_ns = job[rwtype]["clat_ns"]
    return {
        "avg_us": lat_ns["mean"] / 1000,
        "p95_us": lat_ns["percentile"]["95.000000"] / 1000,
        "p99_us": lat_ns["percentile"]["99.000000"] / 1000,
    }

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True,
                        help="Run timestamp from project_manager.py")
    parser.add_argument("--logfile", required=True,
                        help="Unified log file path from project_manager.py")
    parser.add_argument("--reps", type=int, default=3,
                        help="Number of repetitions per workload (default=3)")
    parser.add_argument("--randomize", action="store_true",
                        help="Randomize trial order")
    args = parser.parse_args()

    log_dir, res_dir, fig_dir, ts = prepare_run_dirs("zeroqueue", args.timestamp)
    logfile = Path(args.logfile)

    log_msg("======================================", logfile)
    log_msg(" Starting Zero-Queue Baseline Experiment", logfile)
    log_msg(f" Timestamp: {ts}", logfile)
    log_msg(f" Target file: {TARGET_FILE}", logfile)
    log_msg(f" Repetitions: {args.reps}", logfile)
    log_msg(f" Data pattern: incompressible (randrepeat=0, norandommap, zero_buffers=0)", logfile)
    log_msg("======================================", logfile)

    raw_results = []   # per-rep rows
    summary = []       # aggregated rows

    workloads = randomize_workloads(BS_SETTINGS.copy(), args.randomize, logfile)

    for bs, rw in workloads:
        log_msg(f"Running workload: {bs} {rw}", logfile)
        rep_metrics = []
        for rep in range(1, args.reps + 1):
            out_json = log_dir / f"{bs}_{rw}_rep{rep}.json"
            data = run_fio(bs, rw, out_json, logfile, rep)
            metrics = extract_metrics(data)
            metrics.update({"blocksize": bs, "pattern": rw, "rep": rep})
            raw_results.append(metrics)
            rep_metrics.append(metrics)
            log_msg(f" Rep {rep}: {metrics}", logfile)

        # Aggregate mean and stdev
        def agg(field):
            vals = [m[field] for m in rep_metrics]
            if len(vals) > 1:
                return statistics.mean(vals), statistics.stdev(vals)
            else:
                return vals[0], 0.0

        avg_mean, avg_std = agg("avg_us")
        p95_mean, p95_std = agg("p95_us")
        p99_mean, p99_std = agg("p99_us")

        summary.append({
            "blocksize": bs,
            "pattern": rw,
            "avg_mean_us": avg_mean, "avg_std_us": avg_std,
            "p95_mean_us": p95_mean, "p95_std_us": p95_std,
            "p99_mean_us": p99_mean, "p99_std_us": p99_std,
        })
        log_msg(f" Aggregated: avg={avg_mean:.1f}±{avg_std:.1f}, "
                f"p95={p95_mean:.1f}±{p95_std:.1f}, "
                f"p99={p99_mean:.1f}±{p99_std:.1f}", logfile)

    # Save raw per-rep CSV
    raw_csv = res_dir / "zeroqueue_raw.csv"
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["blocksize", "pattern", "rep", "avg_us", "p95_us", "p99_us"])
        writer.writeheader()
        writer.writerows(raw_results)
    log_msg(f"Raw results CSV written: {raw_csv}", logfile)

    # Save summary CSV
    summary_csv = res_dir / "zeroqueue_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "blocksize", "pattern",
            "avg_mean_us", "avg_std_us",
            "p95_mean_us", "p95_std_us",
            "p99_mean_us", "p99_std_us"
        ])
        writer.writeheader()
        writer.writerows(summary)
    log_msg(f"Summary CSV written: {summary_csv}", logfile)

    # Sort summary for plotting (by block size, then pattern)
    def sort_key(row):
        size = int(row["blocksize"].lower().replace("k", ""))
        pattern_order = {"randread": 0, "read": 0, "randwrite": 1, "write": 1}
        return (size, pattern_order.get(row["pattern"], 99))

    summary.sort(key=sort_key)

    # Plot with error bars
    labels = [f"{r['blocksize']} {r['pattern']}" for r in summary]
    avg = [r["avg_mean_us"] for r in summary]
    p95 = [r["p95_mean_us"] for r in summary]
    p99 = [r["p99_mean_us"] for r in summary]
    avg_err = [r["avg_std_us"] for r in summary]
    p95_err = [r["p95_std_us"] for r in summary]
    p99_err = [r["p99_std_us"] for r in summary]

    x = range(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(x, avg, width=0.25, yerr=avg_err, label="Avg", capsize=5)
    plt.bar([i+0.25 for i in x], p95, width=0.25, yerr=p95_err, label="p95", capsize=5)
    plt.bar([i+0.50 for i in x], p99, width=0.25, yerr=p99_err, label="p99", capsize=5)
    plt.xticks([i+0.25 for i in x], labels, rotation=30)
    plt.ylabel("Latency (µs)")
    plt.title("Zero-Queue Baseline Latencies (QD=1)")
    plt.legend()
    plt.tight_layout()
    fig_file = fig_dir / "zeroqueue.png"
    plt.savefig(fig_file)
    plt.close()
    log_msg(f"Figure saved: {fig_file}", logfile)

    log_msg("Experiment complete.", logfile)
    log_msg("======================================", logfile)

if __name__ == "__main__":
    main()
