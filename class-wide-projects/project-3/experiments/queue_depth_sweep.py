#!/usr/bin/env python3
"""
Queue-Depth / Parallelism Sweep Experiment (numjobs only)

Sweeps concurrency by varying numjobs (with iodepth=1).
Fixed knobs: 4 KiB, random read, working set size configurable, runtime configurable.

Outputs:
- Per-rep CSV
- Summary CSV
- Throughput vs. Latency trade-off curve with error bars
- Knee identified using Kneedle-style geometric method
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
DEFAULT_SIZE = "1G"
DEFAULT_RUNTIME = 30
BLOCK_SIZE = "4k"
RW_MODE = "randread"

# Default sweep: numjobs values
DEFAULT_NUMJOBS = [1, 2, 4, 8, 16, 32]

# ----------------------------
# Helpers
# ----------------------------
def run_fio(numjobs, out_json, logfile, rep_idx, size, runtime):
    """Run fio workload and return parsed JSON."""
    name = f"qd_sweep_nj{numjobs}_rep{rep_idx}"
    cmd = fio_cmd(
        name=name,
        filename=TARGET_FILE,
        rw=RW_MODE,
        bs=BLOCK_SIZE,
        size=size,
        runtime=runtime,
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
    """Extract throughput + latency metrics from fio JSON safely."""
    job = data["jobs"][0]
    section = job["read"]

    iops = section.get("iops", 0)
    mb_s = section.get("bw", 0) / 1024.0  # fio reports KB/s

    # Use total latency (slat + clat)
    lat_ns = section.get("lat_ns", {})
    lat_mean = lat_ns.get("mean", 0)
    lat_p95 = lat_ns.get("percentile", {}).get("95.000000", 0)
    lat_p99 = lat_ns.get("percentile", {}).get("99.000000", 0)

    return {
        "iops": iops,
        "mb_s": mb_s,
        "lat_avg_us": lat_mean / 1000,
        "lat_p95_us": lat_p95 / 1000,
        "lat_p99_us": lat_p99 / 1000,
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
    parser.add_argument("--numjobs", type=int, nargs="+", default=DEFAULT_NUMJOBS,
                        help="List of numjobs values to sweep (default: 1 2 4 8 16 32)")
    parser.add_argument("--size", default=DEFAULT_SIZE,
                        help="Working set size (default=1G)")
    parser.add_argument("--runtime", type=int, default=DEFAULT_RUNTIME,
                        help="Runtime per trial in seconds (default=30)")
    args = parser.parse_args()

    size = args.size
    runtime = args.runtime

    log_dir, res_dir, fig_dir, ts = prepare_run_dirs("queue_depth_sweep", args.timestamp)
    logfile = Path(args.logfile)

    log_msg("======================================", logfile)
    log_msg(" Starting Queue-Depth Sweep Experiment (numjobs only)", logfile)
    log_msg(f" Timestamp: {ts}", logfile)
    log_msg(f" Target file: {TARGET_FILE}", logfile)
    log_msg(f" Numjobs: {args.numjobs}", logfile)
    log_msg(f" Repetitions: {args.reps}", logfile)
    log_msg(f" Size: {size}, Runtime: {runtime}s", logfile)
    log_msg("======================================", logfile)

    raw_results = []
    summary = []

    workloads = randomize_workloads([(nj,) for nj in args.numjobs], args.randomize, logfile)

    for (numjobs,) in workloads:
        conc = numjobs  # since iodepth=1
        log_msg(f"Running workload: numjobs={numjobs}, conc={conc}", logfile)
        rep_metrics = []
        for rep in range(1, args.reps + 1):
            out_json = log_dir / f"nj{numjobs}_rep{rep}.json"
            data = run_fio(numjobs, out_json, logfile, rep, size, runtime)
            metrics = extract_metrics(data)
            metrics.update({"numjobs": numjobs, "concurrency": conc, "rep": rep})
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

        iops_mean, iops_std = agg("iops")
        mb_mean, mb_std = agg("mb_s")
        lat_mean, lat_std = agg("lat_avg_us")

        summary.append({
            "numjobs": numjobs,
            "concurrency": conc,
            "iops_mean": iops_mean, "iops_std": iops_std,
            "mb_mean": mb_mean, "mb_std": mb_std,
            "lat_mean_us": lat_mean, "lat_std_us": lat_std,
        })
        log_msg(f" Aggregated: iops={iops_mean:.1f}±{iops_std:.1f}, "
                f"mb/s={mb_mean:.1f}±{mb_std:.1f}, "
                f"lat={lat_mean:.1f}±{lat_std:.1f} us", logfile)

    # Save raw CSV
    raw_csv = res_dir / "queue_depth_sweep_raw.csv"
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "numjobs", "concurrency", "rep",
            "iops", "mb_s",
            "lat_avg_us", "lat_p95_us", "lat_p99_us"
        ])
        writer.writeheader()
        writer.writerows(raw_results)
    log_msg(f"Raw results CSV written: {raw_csv}", logfile)

    # Save summary CSV
    summary_csv = res_dir / "queue_depth_sweep_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "numjobs", "concurrency",
            "iops_mean", "iops_std",
            "mb_mean", "mb_std",
            "lat_mean_us", "lat_std_us"
        ])
        writer.writeheader()
        writer.writerows(summary)
    log_msg(f"Summary CSV written: {summary_csv}", logfile)

    # ----------------------------
    # Plot throughput vs latency
    # ----------------------------
    summary_sorted = sorted(summary, key=lambda r: r["concurrency"])
    thr = [r["iops_mean"] for r in summary_sorted]
    thr_err = [r["iops_std"] for r in summary_sorted]
    lat = [r["lat_mean_us"] / 1000.0 for r in summary_sorted]  # ms
    lat_err = [r["lat_std_us"] / 1000.0 for r in summary_sorted]
    labels = [f"nj={r['numjobs']}" for r in summary_sorted]

    # ---- Kneedle-style knee detection ----
    knee_idx = None
    if len(lat) > 2:
        x0, y0 = lat[0], thr[0]
        x1, y1 = lat[-1], thr[-1]

        max_dist = -1
        for i in range(len(lat)):
            num = abs((y1 - y0) * lat[i] - (x1 - x0) * thr[i] + x1*y0 - y1*x0)
            den = ((y1 - y0)**2 + (x1 - x0)**2) ** 0.5
            dist = num / den if den > 0 else 0
            if dist > max_dist:
                max_dist = dist
                knee_idx = i

    plt.figure(figsize=(7, 6))
    plt.errorbar(lat, thr, xerr=lat_err, yerr=thr_err,
                 marker="o", linestyle="-", capsize=5, label="numjobs sweep")

    # Annotate each point
    for x, y, label in zip(lat, thr, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Highlight knee point if found
    if knee_idx is not None:
        plt.scatter([lat[knee_idx]], [thr[knee_idx]],
                    color="red", s=100, zorder=5, label="Knee")
        plt.annotate("Knee", (lat[knee_idx], thr[knee_idx]),
                     textcoords="offset points", xytext=(10, -10),
                     ha="left", color="red", fontsize=9)

    plt.xlabel("Average Latency (ms)")
    plt.ylabel("Throughput (IOPS)")
    plt.title("Queue-Depth/Parallelism Sweep: Throughput vs Latency (4 KiB RandRead, numjobs only)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_file = fig_dir / "queue_depth_sweep.png"
    plt.savefig(fig_file)
    plt.close()
    log_msg(f"Figure saved: {fig_file}", logfile)

    log_msg("Experiment complete.", logfile)
    log_msg("======================================", logfile)

if __name__ == "__main__":
    main()
