#!/usr/bin/env python3
"""
Block-Size Sweep Experiment

Sweeps block size from 4 KiB → 256 KiB,
running both sequential and random patterns.
From the same runs, collect IOPS, MB/s, and average latency.

Features:
- Supports repetitions (--reps, default=3)
- Optional randomization of trial order (--randomize)
- Configurable iodepth (--iodepth, default=8)
- Uses incompressible data pattern flags by default
- Logs progress + metrics to unified logfile
- Outputs per-rep CSV, summary CSV, and plots with error bars
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
BLOCK_SIZES = ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]
PATTERNS = {
    "sequential": "read",
    "random": "randread",
}

# ----------------------------
# Helper Functions
# ----------------------------
def run_fio(bs, rw, out_json, logfile, rep_idx, iodepth):
    """Run fio for given workload and return parsed JSON."""
    name = f"blocksweep_{bs}_{rw}_rep{rep_idx}"
    cmd = fio_cmd(
        name=name,
        filename=TARGET_FILE,
        rw=rw,
        bs=bs,
        size=SIZE,
        runtime=RUNTIME,
        iodepth=iodepth,
        numjobs=1,
        out_json=out_json
    )
    log_msg(f"Launching FIO: {name}", logfile)
    log_msg(f"Command: {' '.join(cmd)}", logfile)
    subprocess.run(cmd, check=True)
    log_msg(f"Completed FIO: {name}", logfile)

    with open(out_json) as f:
        return json.load(f)

def extract_metrics(data):
    """Extract throughput (IOPS, MB/s) and latency from fio JSON."""
    job = data["jobs"][0]
    section = job["read"] if job["read"]["io_bytes"] > 0 else job["write"]

    iops = section["iops"]
    mb_s = section["bw"] / 1024.0  # fio reports KB/s
    lat_ns = section["clat_ns"]

    return {
        "iops": iops,
        "mb_s": mb_s,
        "lat_avg_us": lat_ns["mean"] / 1000,
        "lat_p95_us": lat_ns["percentile"]["95.000000"] / 1000,
        "lat_p99_us": lat_ns["percentile"]["99.000000"] / 1000,
    }

def bs_to_kib(bs):
    if bs.endswith("k"):
        return int(bs[:-1])
    if bs.endswith("m"):
        return int(bs[:-1]) * 1024
    return int(bs)

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
    parser.add_argument("--iodepth", type=int, default=8,
                        help="Queue depth for all runs (default=8)")
    args = parser.parse_args()

    log_dir, res_dir, fig_dir, ts = prepare_run_dirs("block_size_sweep", args.timestamp)
    logfile = Path(args.logfile)

    log_msg("======================================", logfile)
    log_msg(" Starting Block-Size Sweep Experiment", logfile)
    log_msg(f" Timestamp: {ts}", logfile)
    log_msg(f" Target file: {TARGET_FILE}", logfile)
    log_msg(f" Block sizes: {BLOCK_SIZES}", logfile)
    log_msg(f" Patterns: {list(PATTERNS.keys())}", logfile)
    log_msg(f" Repetitions: {args.reps}", logfile)
    log_msg(f" iodepth: {args.iodepth}", logfile)
    log_msg("======================================", logfile)

    raw_results = []
    summary = []

    # Iterate both patterns
    for pattern, rw_mode in PATTERNS.items():
        workloads = [(bs, rw_mode) for bs in BLOCK_SIZES]
        workloads = randomize_workloads(workloads, args.randomize, logfile)

        for bs, rw in workloads:
            log_msg(f"Running workload: {pattern} {bs}", logfile)
            rep_metrics = []
            for rep in range(1, args.reps + 1):
                out_json = log_dir / f"{pattern}_{bs}_rep{rep}.json"
                data = run_fio(bs, rw, out_json, logfile, rep, args.iodepth)
                metrics = extract_metrics(data)
                metrics.update({
                    "blocksize": bs,
                    "pattern": pattern,
                    "rw": "read",  # fixed for this assignment
                    "rep": rep
                })
                raw_results.append(metrics)
                rep_metrics.append(metrics)
                log_msg(f" Rep {rep}: {metrics}", logfile)

            # Aggregate
            def agg(field):
                vals = [m[field] for m in rep_metrics]
                return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else (vals[0], 0.0)

            iops_mean, iops_std = agg("iops")
            mb_mean, mb_std = agg("mb_s")
            lat_mean, lat_std = agg("lat_avg_us")

            summary.append({
                "blocksize": bs,
                "pattern": pattern,
                "iops_mean": iops_mean, "iops_std": iops_std,
                "mb_mean": mb_mean, "mb_std": mb_std,
                "lat_mean_us": lat_mean, "lat_std_us": lat_std,
            })
            log_msg(f" Aggregated: iops={iops_mean:.1f}±{iops_std:.1f}, "
                    f"mb/s={mb_mean:.1f}±{mb_std:.1f}, "
                    f"lat={lat_mean:.1f}±{lat_std:.1f} us", logfile)

    # Save raw per-rep CSV
    raw_csv = res_dir / "block_size_sweep_raw.csv"
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "blocksize", "pattern", "rw", "rep",
            "iops", "mb_s", "lat_avg_us", "lat_p95_us", "lat_p99_us"
        ])
        writer.writeheader()
        writer.writerows(raw_results)
    log_msg(f"Raw results CSV written: {raw_csv}", logfile)

    # Save summary CSV
    summary_csv = res_dir / "block_size_sweep_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "blocksize", "pattern",
            "iops_mean", "iops_std",
            "mb_mean", "mb_std",
            "lat_mean_us", "lat_std_us"
        ])
        writer.writeheader()
        writer.writerows(summary)
    log_msg(f"Summary CSV written: {summary_csv}", logfile)

    # Plot per pattern
    for pattern in PATTERNS.keys():
        summary_pat = [r for r in summary if r["pattern"] == pattern]
        summary_pat = sorted(summary_pat, key=lambda r: bs_to_kib(r["blocksize"]))
        sizes = [r["blocksize"] for r in summary_pat]
        x = range(len(sizes))

        iops = [r["iops_mean"] for r in summary_pat]
        iops_err = [r["iops_std"] for r in summary_pat]
        mb = [r["mb_mean"] for r in summary_pat]
        mb_err = [r["mb_std"] for r in summary_pat]
        lat = [r["lat_mean_us"] / 1000.0 for r in summary_pat]  # ms

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # Throughput panel
        ax1.errorbar(x, iops, yerr=iops_err, marker="o", label="IOPS")
        ax1_twin = ax1.twinx()
        ax1_twin.errorbar(x, mb, yerr=mb_err, marker="s", color="orange", label="MB/s")

        ax1.set_ylabel("IOPS")
        ax1_twin.set_ylabel("MB/s")
        ax1.set_title(f"Block Size Sweep - {pattern}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(sizes, rotation=30)

        # Crossover markers (64 KiB → 128 KiB)
        for cutoff in ["64k", "128k"]:
            if cutoff in sizes:
                idx = sizes.index(cutoff)
                ax1.axvline(idx, color="gray", linestyle="--", linewidth=1)
        ax1.text(0.5, 0.9, "IOPS focus ≤64K\nMB/s focus ≥128K",
                 transform=ax1.transAxes, ha="center", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")

        # Latency panel
        ax2.plot(x, lat, marker="o", label="Avg Latency (ms)")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_xlabel("Block Size")
        ax2.legend()

        plt.tight_layout()
        fig_file = fig_dir / f"block_size_sweep_{pattern}.png"
        plt.savefig(fig_file)
        plt.close()
        log_msg(f"Figure saved: {fig_file}", logfile)

    log_msg("Experiment complete.", logfile)
    log_msg("======================================", logfile)

if __name__ == "__main__":
    main()
