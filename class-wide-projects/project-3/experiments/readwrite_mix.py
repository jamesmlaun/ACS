#!/usr/bin/env python3
"""
Read/Write Mix Sweep Experiment

Fixed knobs: 4 KiB, random pattern, 1 GiB size
Runs four mixes:
- 100% Reads
- 100% Writes
- 70/30 Read/Write
- 50/50 Read/Write

Features:
- Supports repetitions (--reps, default=3)
- Optional randomization of trial order (--randomize)
- Configurable iodepth (--iodepth, default=8)
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
BLOCK_SIZE = "4k"

MIXES = [
    ("100R", "randread", None),        # 100% read
    ("100W", "randwrite", None),       # 100% write
    ("70/30", "randrw", 70),           # 70% read
    ("50/50", "randrw", 50),           # 50% read
]

# ----------------------------
# Helpers
# ----------------------------
def run_fio(mix_label, rw, rwmixread, out_json, logfile, rep_idx, iodepth):
    """Run fio workload and return parsed JSON."""
    safe_label = mix_label.replace("/", "-")  # avoid slashes in filenames
    name = f"rw_mix_{safe_label}_rep{rep_idx}"
    cmd = fio_cmd(
        name=name,
        filename=TARGET_FILE,
        rw=rw,
        bs=BLOCK_SIZE,
        size=SIZE,
        runtime=RUNTIME,
        iodepth=iodepth,
        numjobs=1,
        out_json=out_json
    )
    if rwmixread is not None:
        cmd.insert(-2, f"--rwmixread={rwmixread}")  # before output flags

    log_msg(f"Launching FIO: {name}", logfile)
    subprocess.run(cmd, check=True)
    log_msg(f"Completed FIO: {name}", logfile)

    with open(out_json) as f:
        return json.load(f)

def extract_metrics(data):
    """Extract throughput + latency metrics from fio JSON safely."""
    job = data["jobs"][0]
    read = job["read"]
    write = job["write"]

    total_iops = read.get("iops", 0) + write.get("iops", 0)
    total_mb = (read.get("bw", 0) + write.get("bw", 0)) / 1024.0  # fio bw in KB/s

    # Weighted latency
    total_ios = read.get("total_ios", 0) + write.get("total_ios", 0)
    if total_ios > 0:
        lat_mean = (
            read.get("clat_ns", {}).get("mean", 0) * read.get("total_ios", 0)
            + write.get("clat_ns", {}).get("mean", 0) * write.get("total_ios", 0)
        ) / total_ios

        def get_pct(section, pct):
            return section.get("clat_ns", {}).get("percentile", {}).get(pct, 0)

        lat_p95 = max(get_pct(read, "95.000000"), get_pct(write, "95.000000"))
        lat_p99 = max(get_pct(read, "99.000000"), get_pct(write, "99.000000"))
    else:
        lat_mean = lat_p95 = lat_p99 = 0

    return {
        "iops": total_iops,
        "mb_s": total_mb,
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
    parser.add_argument("--iodepth", type=int, default=8,
                        help="Queue depth (default=8)")
    args = parser.parse_args()

    log_dir, res_dir, fig_dir, ts = prepare_run_dirs("readwrite_mix", args.timestamp)
    logfile = Path(args.logfile)

    log_msg("======================================", logfile)
    log_msg(" Starting Read/Write Mix Sweep", logfile)
    log_msg(f" Timestamp: {ts}", logfile)
    log_msg(f" Target file: {TARGET_FILE}", logfile)
    log_msg(f" Block size: {BLOCK_SIZE}", logfile)
    log_msg(f" Runtime: {RUNTIME}s", logfile)
    log_msg(f" Mixes: {[m[0] for m in MIXES]}", logfile)
    log_msg(f" Repetitions: {args.reps}", logfile)
    log_msg(f" iodepth: {args.iodepth}", logfile)
    log_msg("======================================", logfile)

    raw_results = []
    summary = []

    workloads = randomize_workloads(MIXES.copy(), args.randomize, logfile)

    for mix_label, rw, rwmixread in workloads:
        log_msg(f"Running workload: {mix_label}", logfile)
        rep_metrics = []
        for rep in range(1, args.reps + 1):
            out_json = log_dir / f"{mix_label.replace('/', '-')}_rep{rep}.json"
            data = run_fio(mix_label, rw, rwmixread, out_json, logfile, rep, args.iodepth)
            metrics = extract_metrics(data)
            metrics.update({"mix": mix_label, "rw": rw, "rep": rep})
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
            "mix": mix_label,
            "iops_mean": iops_mean, "iops_std": iops_std,
            "mb_mean": mb_mean, "mb_std": mb_std,
            "lat_mean_us": lat_mean, "lat_std_us": lat_std,
        })
        log_msg(f" Aggregated: iops={iops_mean:.1f}±{iops_std:.1f}, "
                f"mb/s={mb_mean:.1f}±{mb_std:.1f}, "
                f"lat={lat_mean:.1f}±{lat_std:.1f} us", logfile)

    # Save raw CSV
    raw_csv = res_dir / "readwrite_mix_raw.csv"
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mix", "rw", "rep",
            "iops", "mb_s",
            "lat_avg_us", "lat_p95_us", "lat_p99_us"
        ])
        writer.writeheader()
        writer.writerows(raw_results)
    log_msg(f"Raw results CSV written: {raw_csv}", logfile)

    # Save summary CSV
    summary_csv = res_dir / "readwrite_mix_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mix",
            "iops_mean", "iops_std",
            "mb_mean", "mb_std",
            "lat_mean_us", "lat_std_us"
        ])
        writer.writeheader()
        writer.writerows(summary)
    log_msg(f"Summary CSV written: {summary_csv}", logfile)

    # ----------------------------
    # Plot (IOPS only + latency)
    # ----------------------------
    order = {"100R": 100, "70/30": 70, "50/50": 50, "100W": 0}
    summary_sorted = sorted(summary, key=lambda r: order[r["mix"]])

    mixes = [r["mix"] for r in summary_sorted]
    x = range(len(mixes))
    iops = [r["iops_mean"] for r in summary_sorted]
    iops_err = [r["iops_std"] for r in summary_sorted]
    lat = [r["lat_mean_us"] / 1000.0 for r in summary_sorted]  # ms
    lat_err = [r["lat_std_us"] / 1000.0 for r in summary_sorted]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Throughput (IOPS only)
    ax1.errorbar(x, iops, yerr=iops_err, marker="o", label="IOPS", color="tab:blue")
    ax1.set_ylabel("IOPS")
    ax1.set_title("Read/Write Mix Sweep (4 KiB Random)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mixes, rotation=30)
    ax1.legend(loc="best")
    ax1.text(0.5, 0.02, "Note: MiB/s = IOPS × 0.00390625 (4 KiB)",
             transform=ax1.transAxes, ha="center", va="bottom", fontsize=9, alpha=0.8)

    # Latency
    ax2.errorbar(x, lat, yerr=lat_err, marker="o", color="tab:green", label="Avg Latency (ms)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_xlabel("Read/Write Mix")
    ax2.legend()

    plt.tight_layout()
    fig_file = fig_dir / "readwrite_mix.png"
    plt.savefig(fig_file)
    plt.close()
    log_msg(f"Figure saved: {fig_file}", logfile)

    log_msg("Experiment complete.", logfile)
    log_msg("======================================", logfile)

if __name__ == "__main__":
    main()
