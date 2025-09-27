"""
Experiment 4: Intensity Sweep (Intel MLC v3.11b)
- Uses --loaded_latency with specific inject delays (-d) per intensity
- Collects throughput (MB/s) and latency (ns)
- CSV schema: rep,intensity,latency_ns,bandwidth_MBps
- Averages across reps; plots throughput vs latency with x/y error bars
- Identifies "knee" point and explains with Little's Law
- Computes theoretical peak bandwidth and % achieved
- Draws theoretical peak line on plot
"""

import subprocess
import csv
import time
import random
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
from experiments import utils

MLC_PATH   = str(Path.home() / "mlc")
BUFFER_SIZE = "256M"   # DRAM region
LINE_LEN    = 64       # bytes/request used by MLC for each access (-l64)

# Intensities: inject delay values (MLC arg to --loaded_latency)
INTENSITIES = {
    "very_low":  "800",
    "low":       "400",
    "medium":    "200",
    "high":      "100",
    "very_high": "50",
}

# ---- Theoretical peak bandwidth parameters ----
CHANNELS = 2            # dual channel
DATA_RATE_MT_S = 3200   # DDR4-3200 => 3200 MT/s
BUS_WIDTH_BYTES = 8     # 64-bit channel = 8 bytes

def theoretical_peak_GBs(channels=CHANNELS,
                         bus_bytes=BUS_WIDTH_BYTES,
                         rate=DATA_RATE_MT_S) -> float:
    """Compute theoretical DRAM peak bandwidth in GB/s (decimal)."""
    return channels * bus_bytes * rate / 1e3  # (MT/s × bytes) → MB/s; /1e3 → GB/s


def _parse_loaded_latency_first_row(text: str):
    """Parse latency (ns) and bandwidth (MB/s) from --loaded_latency output."""
    for line in utils.filter_and_log(text):
        toks = line.strip().split()
        if toks and toks[0].isdigit() and len(toks) >= 3:
            try:
                lat_ns = float(toks[1])
                bw     = float(toks[2])
                return lat_ns, bw
            except ValueError:
                continue
    return None, None


def _find_knee(latencies_ns, throughputs_MBps):
    """
    Kneedle-style knee detection:
      - Normalize data by sorting in ascending latency.
      - Compute distance of each point to the line between the first and last point.
      - Return the index (into the *sorted* list) of the point with maximum distance.
    """
    if len(latencies_ns) < 3:
        return None

    # Sort points by latency
    order = sorted(range(len(latencies_ns)), key=lambda i: latencies_ns[i])
    lats = [latencies_ns[i] for i in order]
    bws  = [throughputs_MBps[i] for i in order]

    # Line endpoints (first and last point)
    x1, y1 = lats[0], bws[0]
    x2, y2 = lats[-1], bws[-1]

    # Precompute line vector and length
    dx, dy = x2 - x1, y2 - y1
    denom = (dx**2 + dy**2) ** 0.5
    if denom == 0:
        return None

    # Find point with max perpendicular distance to the line
    max_dist, knee_sorted_idx = -1, None
    for i in range(1, len(lats) - 1):  # exclude endpoints
        x0, y0 = lats[i], bws[i]
        dist = abs(dy*x0 - dx*y0 + x2*y1 - y2*x1) / denom
        if dist > max_dist:
            max_dist, knee_sorted_idx = dist, i

    # Map back to original index space
    return order[knee_sorted_idx] if knee_sorted_idx is not None else None



def _effective_outstanding_requests(bw_MBps, lat_ns, bytes_per_req=LINE_LEN):
    """Little's Law: L ≈ (BW_Bps * W_sec) / bytes_per_req"""
    bw_Bps = bw_MBps * 1_000_000.0
    w_sec  = lat_ns * 1e-9
    return (bw_Bps * w_sec) / bytes_per_req if bytes_per_req>0 else float("nan")


def run(reps: int = 3, warmup: bool = True, randomize: bool = True):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")

    logs_dir    = Path("logs")
    results_dir = Path("results") / f"intensity_sweep_{ts}"
    figures_dir = Path("figures") / f"intensity_sweep_{ts}"
    for d in (logs_dir, results_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_file     = logs_dir / f"intensity_sweep_{ts}.log"
    raw_csv      = results_dir / "raw_data.csv"
    summary_csv  = results_dir / "summary.csv"
    knee_txt     = results_dir / "knee.txt"
    peak_txt     = results_dir / "peak.txt"
    plot_file    = figures_dir / "throughput_latency.png"

    print(f"[INFO] Intensity Sweep (reps={reps}, warmup={warmup}, randomize={randomize})")

    # intensity label -> list of (lat, bw)
    samples = {k: [] for k in INTENSITIES.keys()}

    with open(log_file, "w") as log, open(raw_csv, "w", newline="") as raw_out:
        writer = csv.writer(raw_out)
        writer.writerow(["rep", "intensity", "latency_ns", "bandwidth_MBps"])
        utils.log_environment(log, MLC_PATH)

        # Warm-up
        if warmup:
            log.write("=== Warm-up Phase (discarded) ===\n")
            for lbl, inject in INTENSITIES.items():
                cmd = ["taskset","-c","0",MLC_PATH,"--loaded_latency",
                       f"-b{BUFFER_SIZE}",f"-l{LINE_LEN}",f"-d{inject}"]
                log.write(f"[WARMUP] {' '.join(cmd)}\n")
                subprocess.run(cmd,capture_output=True,text=True)
            log.write("=== End Warm-up ===\n\n")

        configs = list(INTENSITIES.items())
        for rep in range(1,reps+1):
            tests = configs[:]
            if randomize: random.shuffle(tests)
            for lbl, inject in tests:
                print(f"[INFO] rep {rep}/{reps} intensity={lbl} (inject={inject})")
                cmd = ["taskset","-c","0",MLC_PATH,"--loaded_latency",
                       f"-b{BUFFER_SIZE}",f"-l{LINE_LEN}",f"-d{inject}"]
                log.write(f"[RUN] rep={rep}, intensity={lbl}\n")
                log.write("Command: "+" ".join(cmd)+"\n")
                res = subprocess.run(cmd,capture_output=True,text=True)
                for line in utils.filter_and_log(res.stdout): log.write(line+"\n")
                lat,bw = _parse_loaded_latency_first_row(res.stdout)
                if lat is not None and bw is not None:
                    samples[lbl].append((lat,bw))
                    writer.writerow([rep,lbl,f"{lat:.2f}",f"{bw:.2f}"])

    # ---- Summaries ----
    labels = list(INTENSITIES.keys())
    lat_means, lat_stds, bw_means, bw_stds = [],[],[],[]
    for lbl in labels:
        vals = samples.get(lbl,[])
        if vals:
            lats=[v[0] for v in vals]; bws=[v[1] for v in vals]
            lat_means.append(statistics.mean(lats))
            lat_stds.append(statistics.stdev(lats) if len(lats)>1 else 0.0)
            bw_means.append(statistics.mean(bws))
            bw_stds.append(statistics.stdev(bws) if len(bws)>1 else 0.0)
        else:
            lat_means.append(float("nan")); lat_stds.append(float("nan"))
            bw_means.append(float("nan"));  bw_stds.append(float("nan"))

    # Theoretical peak + measured peak
    peak_theoretical = theoretical_peak_GBs()
    bw_meas_max = max([bw for bw in bw_means if bw==bw], default=float("nan"))
    bw_meas_max_GBs = bw_meas_max/1000.0
    pct_of_peak = (100.0*bw_meas_max_GBs/peak_theoretical
                   if peak_theoretical>0 else float("nan"))

    with open(summary_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["intensity","lat_mean_ns","lat_std_ns","bw_mean_MBps","bw_std_MBps"])
        for i,lbl in enumerate(labels):
            w.writerow([lbl,
                        f"{lat_means[i]:.2f}" if lat_means[i]==lat_means[i] else "nan",
                        f"{lat_stds[i]:.2f}"  if lat_stds[i]==lat_stds[i]  else "nan",
                        f"{bw_means[i]:.2f}"  if bw_means[i]==bw_means[i]  else "nan",
                        f"{bw_stds[i]:.2f}"   if bw_stds[i]==bw_stds[i]   else "nan"])
        w.writerow([])
        w.writerow(["Theoretical Peak GB/s", f"{peak_theoretical:.2f}"])
        w.writerow(["Measured Peak GB/s", f"{bw_meas_max_GBs:.2f}"])
        w.writerow(["Percent of Peak", f"{pct_of_peak:.1f}%"])

    with open(peak_txt,"w") as f:
        f.write(f"Theoretical Peak: {peak_theoretical:.2f} GB/s\n")
        f.write(f"Measured Peak: {bw_meas_max_GBs:.2f} GB/s\n")
        f.write(f"Percent of Peak: {pct_of_peak:.1f}%\n")

    # ---- Plot (sorted for smooth curve) ----
    idx_sorted = sorted([i for i in range(len(labels)) if lat_means[i]==lat_means[i]],
                        key=lambda i: lat_means[i])
    lats=[lat_means[i] for i in idx_sorted]; lerr=[lat_stds[i] for i in idx_sorted]
    bws=[bw_means[i] for i in idx_sorted];  berr=[bw_stds[i] for i in idx_sorted]
    labs=[labels[i] for i in idx_sorted]

    plt.figure(figsize=(8,6))
    plt.errorbar(lats,bws,xerr=lerr,yerr=berr,
                 fmt="o-",capsize=5,elinewidth=1.5,markersize=6)
    for i,txt in enumerate(labs):
        plt.annotate(txt,(lats[i],bws[i]),textcoords="offset points",xytext=(6,6))

    # ---- Theoretical peak line ----
    plt.axhline(peak_theoretical*1000, color="blue", linestyle="--", alpha=0.5,
                label=f"Theoretical Peak ({peak_theoretical:.1f} GB/s)")
    # (×1000 because measured BW is in MB/s)

    # ---- Knee detection + Little's Law ----
    knee_local=_find_knee(lats,bws)
    if knee_local is not None:
        knee_lat=lats[knee_local]; knee_bw=bws[knee_local]
        knee_L=_effective_outstanding_requests(knee_bw,knee_lat,LINE_LEN)
        peak_bw=max(bws) if bws else float("nan")
        pct_peak_meas=(100.0*knee_bw/peak_bw if peak_bw and peak_bw==peak_bw else float("nan"))

        plt.scatter([knee_lat],[knee_bw],s=90,zorder=5,color="red",label="Knee")
        plt.axvline(knee_lat,color="red",linestyle="--",alpha=0.35)
        plt.axhline(knee_bw,color="red",linestyle="--",alpha=0.35)

        with open(knee_txt,"w") as f:
            f.write("KNEE IDENTIFICATION (heuristic: largest slope drop)\n")
            f.write(f"knee_latency_ns={knee_lat:.2f}\n")
            f.write(f"knee_throughput_MBps={knee_bw:.2f}\n")
            f.write(f"knee_effective_outstanding_requests≈{knee_L:.2f}\n")
            f.write(f"knee_percent_of_measured_peak≈{pct_peak_meas:.1f}%\n\n")
            f.write("Little's Law tie-in:\n")
            f.write("  L = λ·W, with λ ≈ BW/bytes_per_request.\n")
            f.write(f"  Here bytes/request ≈ {LINE_LEN} B.\n")
            f.write("  Before the knee, increasing intensity raises λ (higher BW) while W ~ service time.\n")
            f.write("  After the knee, BW saturates; further intensity inflates W (queuing),\n")
            f.write("  so L grows mostly via latency, showing diminishing returns.\n")

    plt.xlabel("Latency (ns)")
    plt.ylabel("Throughput (MB/s)")
    plt.title("Intensity Sweep: Throughput vs. Latency")
    plt.grid(True,linestyle="--",alpha=0.7)
    plt.legend()
    plt.savefig(plot_file,dpi=300,bbox_inches="tight")
    plt.close()

    print(f"[DONE] Results:\n - {raw_csv}\n - {summary_csv}\n - {peak_txt}\n - {knee_txt}\n - {plot_file}\n - {log_file}")
