#!/usr/bin/env python3
"""
Tail Latency Characterization Experiment

Workload:
- 4 KiB random read
- Test at mid-QD (numjobs=4) and near-knee QD (numjobs=8)

Features:
- Supports repetitions (--reps, default=3)
- Optional randomization (--randomize)
- Captures latency percentiles (avg, p95, p99 by default; p50 and p99.9 if available)
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
# Defaults
# ----------------------------
SIZE = "1G"
RUNTIME = 30
BLOCK_SIZE = "4k"
RW_MODE = "randread"

# Fixed queue depths for tail latency experiment
TAIL_QDS = [4, 8]   # mid-QD and knee-QD

# ----------------------------
# Helpers
# ----------------------------
def run_fio(numjobs, out_json, logfile, rep_idx):
    """Run fio workload and return parsed JSON."""
    name = f"taillat_nj{numjobs}_rep{rep_idx}"
    cmd = fio_cmd(
        name=name,
        filename=TARGET_FILE,
        rw=RW_MODE,
        bs=BLOCK_SIZE,
        size=SIZE,
        runtime=RUNTIME,
        iodepth=1,
        numjobs=numjobs,
        out_json=out_json
    )
    log_msg(f"Launching FIO: {name}", logfile)
    subprocess.run(cmd, check=True)
    log_msg(f"Completed FIO: {name}", logfile)

    with open(out_json) as f:
        return json.load(f)

def extract_metrics(data):
    """Extract avg, p95, p99 (always present) and p50, p99.9 if available."""
    job = data["jobs"][0]
    read = job["read"]

    lat_ns = read.get("clat_ns", {})
    pct = lat_ns.get("percentile", {})

    return {
        "avg_us": lat_ns.get("mean", 0) / 1000,
        "p50_us": pct.get("50.000000", 0) / 1000,
        "p95_us": pct.get("95.000000", 0) / 1000,
        "p99_us": pct.get("99.000000", 0) / 1000,
        "p99_9_us": pct.get("99.900000", 0) / 1000,
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

    log_dir, res_dir, fig_dir, ts = prepare_run_dirs("tail_latency", args.timestamp)
    logfile = Path(args.logfile)

    log_msg("======================================", logfile)
    log_msg(" Starting Tail Latency Characterization", logfile)
    log_msg(f" Timestamp: {ts}", logfile)
    log_msg(f" Target file: {TARGET_FILE}", logfile)
    log_msg(f" Queue depths tested: {TAIL_QDS} (mid-QD, knee-QD)", logfile)
    log_msg(f" Block size: {BLOCK_SIZE}, Pattern: {RW_MODE}", logfile)
    log_msg(f" Repetitions: {args.reps}", logfile)
    log_msg("======================================", logfile)

    raw_results = []
    summary = []

    workloads = randomize_workloads([(qd,) for qd in TAIL_QDS], args.randomize, logfile)

    for (numjobs,) in workloads:
        log_msg(f"Running workload: numjobs={numjobs}", logfile)
        rep_metrics = []
        for rep in range(1, args.reps + 1):
            out_json = log_dir / f"nj{numjobs}_rep{rep}.json"
            data = run_fio(numjobs, out_json, logfile, rep)
            metrics = extract_metrics(data)
            metrics.update({"numjobs": numjobs, "rep": rep})
            raw_results.append(metrics)
            rep_metrics.append(metrics)
            log_msg(f" Rep {rep}: {metrics}", logfile)

        # Aggregate
        def agg(field):
            vals = [m[field] for m in rep_metrics]
            if len(vals) > 1:
                return statistics.mean(vals), statistics.stdev(vals)
            else:
                return vals[0], 0.0

        avg_mean, avg_std = agg("avg_us")
        p50_mean, p50_std = agg("p50_us")
        p95_mean, p95_std = agg("p95_us")
        p99_mean, p99_std = agg("p99_us")
        p999_mean, p999_std = agg("p99_9_us")

        summary.append({
            "numjobs": numjobs,
            "avg_mean_us": avg_mean, "avg_std_us": avg_std,
            "p50_mean_us": p50_mean, "p50_std_us": p50_std,
            "p95_mean_us": p95_mean, "p95_std_us": p95_std,
            "p99_mean_us": p99_mean, "p99_std_us": p99_std,
            "p99_9_mean_us": p999_mean, "p99_9_std_us": p999_std,
        })

        log_msg(
            f" Aggregated: avg={avg_mean:.1f}±{avg_std:.1f}, "
            f"p50={p50_mean:.1f}±{p50_std:.1f}, "
            f"p95={p95_mean:.1f}±{p95_std:.1f}, "
            f"p99={p99_mean:.1f}±{p99_std:.1f}, "
            f"p99.9={p999_mean:.1f}±{p999_std:.1f}",
            logfile
        )

    # Save raw CSV
    raw_csv = res_dir / "tail_latency_raw.csv"
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "numjobs", "rep", "avg_us", "p50_us", "p95_us", "p99_us", "p99_9_us"
        ])
        writer.writeheader()
        writer.writerows(raw_results)
    log_msg(f"Raw results CSV written: {raw_csv}", logfile)

    # Save summary CSV
    summary_csv = res_dir / "tail_latency_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "numjobs",
            "avg_mean_us", "avg_std_us",
            "p50_mean_us", "p50_std_us",
            "p95_mean_us", "p95_std_us",
            "p99_mean_us", "p99_std_us",
            "p99_9_mean_us", "p99_9_std_us",
        ])
        writer.writeheader()
        writer.writerows(summary)
    log_msg(f"Summary CSV written: {summary_csv}", logfile)

    # ----------------------------
    # Plot
    # ----------------------------
    label_map = {
        4: "mid-QD (nj=4)",
        8: "near-knee QD (nj=8)"
    }
    labels = [label_map.get(r['numjobs'], f"nj={r['numjobs']}") for r in summary]

    avg = [r["avg_mean_us"] for r in summary]
    p95 = [r["p95_mean_us"] for r in summary]
    p99 = [r["p99_mean_us"] for r in summary]

    avg_err = [r["avg_std_us"] for r in summary]
    p95_err = [r["p95_std_us"] for r in summary]
    p99_err = [r["p99_std_us"] for r in summary]

    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 6))
    plt.bar([i - width for i in x], avg, width, yerr=avg_err, capsize=5, label="Avg")
    plt.bar(x, p95, width, yerr=p95_err, capsize=5, label="p95")
    plt.bar([i + width for i in x], p99, width, yerr=p99_err, capsize=5, label="p99")

    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Latency (µs)")
    plt.title("Tail Latency Characterization (4 KiB RandRead)")
    plt.legend()
    plt.tight_layout()

    fig_file = fig_dir / "tail_latency.png"
    plt.savefig(fig_file)
    plt.close()
    log_msg(f"Figure saved: {fig_file}", logfile)

    log_msg("Experiment complete.", logfile)
    log_msg("======================================", logfile)

if __name__ == "__main__":
    main()
