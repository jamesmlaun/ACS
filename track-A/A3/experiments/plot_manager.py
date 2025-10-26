#!/usr/bin/env python3
# ============================================================
# plot_manager.py — Unified plotting suite for ACS Project A3
# ============================================================
# Purpose:
#   Handles all four experiment visualizations with support for
#   multi-trial averaging (summary.csv) and error bars (mean ± std).
#
# Behavior:
#   - Automatically detects *_mean / *_std columns in summary.csv.
#   - Falls back to single-run results.csv if summary.csv absent.
#   - Generates labeled, publication-ready plots using matplotlib.
#   - Compatible with pandas ≥ 2.2 and numpy ≥ 1.26.
#
# Experiments handled:
#   1. space_vs_accuracy
#   2. lookup_latency
#   3. insert_delete
#   4. thread_scaling
#
# Output:
#   results/<experiment>/<timestamp>/plots/*.png
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# ------------------------------------------------------------
# Global paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def _latest_result_dir(exp_name: str) -> Path:
    """
    Return the latest timestamped results directory for the experiment.
    Raises if not found.
    """
    exp_root = RESULTS_DIR / exp_name
    if not exp_root.exists():
        raise FileNotFoundError(f"No results directory for '{exp_name}'.")
    subdirs = [d for d in exp_root.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No timestamped results found under {exp_root}")
    return max(subdirs, key=lambda d: d.stat().st_mtime)


def _load_results(exp_name: str) -> pd.DataFrame:
    """
    Load summary.csv if present, else results.csv.
    """
    run_dir = _latest_result_dir(exp_name)
    csv_summary = run_dir / "summary.csv"
    csv_results = run_dir / "results.csv"
    if csv_summary.exists():
        print(f"[INFO] Loaded summary (mean/std) from {csv_summary}")
        df = pd.read_csv(csv_summary)
    elif csv_results.exists():
        print(f"[INFO] Loaded raw results from {csv_results}")
        df = pd.read_csv(csv_results)
    else:
        raise FileNotFoundError(f"No results.csv or summary.csv found for {exp_name}.")
    return df, run_dir


def _ensure_plot_dir(run_dir: Path) -> Path:
    """Ensure plots subdirectory exists."""
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _has_aggregate(df: pd.DataFrame) -> bool:
    """Return True if DataFrame includes mean/std columns."""
    return any(col.endswith("_mean") for col in df.columns)


def _parse_hist_series(series):
    hists = []
    for s in series.dropna().astype(str):
        raw = s.strip().strip('"')
        if not raw:
            continue
        try:
            vals = [float(x) for x in raw.split(",") if x.strip() != ""]
            hists.append(np.array(vals, dtype=float))
        except Exception:
            continue
    return hists


def _average_hists(hists):
    if not hists:
        return None
    max_len = max(len(h) for h in hists)
    aligned = [np.pad(h, (0, max_len - len(h)), constant_values=0.0) for h in hists]
    mean_hist = np.mean(np.stack(aligned, axis=0), axis=0)
    total = float(mean_hist.sum())
    return (mean_hist / total) if total > 0 else mean_hist


# ------------------------------------------------------------
# 1. Space vs Accuracy
# ------------------------------------------------------------
def plot_space_vs_accuracy():
    df, run_dir = _load_results("space_vs_accuracy")
    plot_dir = _ensure_plot_dir(run_dir)

    plt.figure(figsize=(6.5, 4))
    plt.title("Space vs Accuracy")
    plt.xlabel("Achieved False Positive Rate")
    plt.ylabel("Bits per Entry (incl. metadata)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    use_mean = _has_aggregate(df)

    for f in sorted(df["filter"].unique()):
        if use_mean:
            sub = df[df["filter"] == f].sort_values("achieved_fpr_mean")
            x = np.asarray(sub["achieved_fpr_mean"])
            y = np.asarray(sub["bpe_mean"])
            xerr = np.asarray(sub.get("achieved_fpr_std", 0))
            yerr = np.asarray(sub.get("bpe_std", 0))
            plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o-", capsize=3, label=f)
        else:
            sub = df[df["filter"] == f].sort_values("achieved_fpr")
            x = np.asarray(sub["achieved_fpr"])
            y = np.asarray(sub["bpe"])
            plt.plot(x, y, "o-", label=f)

    # Theoretical Bloom line
    theory_p = np.logspace(-3, -0.3, 200)
    theory_bpe = -np.log(theory_p) / (np.log(2) ** 2)
    plt.plot(theory_p, theory_bpe, "k--", linewidth=1.0, label="Bloom (theory)")

    plt.legend()
    plt.tight_layout()
    out_path = plot_dir / "space_vs_accuracy.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {out_path}")


# ------------------------------------------------------------
# 2. Lookup Throughput & Latency Tails
# ------------------------------------------------------------
def plot_lookup_latency():
    import numpy as np
    import matplotlib.pyplot as plt

    df, run_dir = _load_results("lookup_latency")
    plot_dir = _ensure_plot_dir(run_dir)
    use_mean = _has_aggregate(df)

    # ---------------------------
    # Throughput (QPS)
    # ---------------------------
    plt.figure(figsize=(8, 5))
    plt.title("Lookup Throughput vs Negative Lookup Share")
    plt.xlabel("Negative Lookup Share")
    plt.ylabel("Queries per second (QPS)")
    plt.grid(True, linestyle="--", alpha=0.4)

    for f in sorted(df["filter"].unique()):
        sub = df[df["filter"] == f].sort_values("neg_ratio")
        if sub.empty:
            continue
        x = sub["neg_ratio"].to_numpy()
        y = sub["qps_mean"].to_numpy() if use_mean else sub["qps"].to_numpy()
        yerr = sub["qps_std"].to_numpy() if "qps_std" in sub.columns else None
        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, label=f)
    plt.legend()
    plt.tight_layout()
    out_tp = plot_dir / "lookup_throughput.png"
    plt.savefig(out_tp, dpi=150)
    plt.close()

    # ---------------------------
    # Latency subplots with FPR annotation
    # ---------------------------
    filters = sorted(df["filter"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = {"p50_ns": "tab:blue", "p95_ns": "tab:orange", "p99_ns": "tab:red"}

    ymax = 0.0
    for idx, f in enumerate(filters):
        ax = axes[idx]
        sub = df[df["filter"] == f].sort_values("neg_ratio")
        if sub.empty:
            ax.set_visible(False)
            continue

        x = sub["neg_ratio"].to_numpy()
        for p in ["p50_ns", "p95_ns", "p99_ns"]:
            col = f"{p}_mean" if use_mean and f"{p}_mean" in sub.columns else p
            y = sub[col].to_numpy()
            ax.plot(x, y, "o-", label=p.replace("_ns", "").upper(), color=colors[p])
            if np.isfinite(y).any():
                ymax = max(ymax, np.nanmax(y))

        # Annotate achieved FPR near p99 peak
        if "achieved_fpr_mean" in sub.columns:
            peak_row = sub.loc[sub["p99_ns_mean"].idxmax()]
            fpr_val = peak_row["achieved_fpr_mean"]
            peak_lat = peak_row["p99_ns_mean"]
            ax.text(
                0.5,
                peak_lat * 1.05,
                f"FPR={fpr_val:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="red",
            )

        ax.set_title(f"{f.capitalize()} Filter", fontsize=10, pad=4)
        ax.set_xlabel("Negative Lookup Share")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx % 2 == 0:
            ax.set_ylabel("Latency (ns)")
        ax.legend(fontsize="small", loc="upper left")

    for ax in axes:
        ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)

    plt.suptitle("Lookup Latency Tails per Filter (p50/p95/p99, with FPR)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_lat = plot_dir / "lookup_latency_subplots.png"
    plt.savefig(out_lat, dpi=150)
    plt.close()

    print(f"[PLOT] Saved throughput → {out_tp.name} and latency → {out_lat.name} in {plot_dir}")


# ------------------------------------------------------------
# 3. Insert/Delete Throughput (Dynamic Filters)
# ------------------------------------------------------------
def plot_insert_delete():
    """
    Unified insert/delete experiment plot generator (ACS A3).
    Produces exactly five figures:
      1. insert_delete.png             – insert & delete throughput for both filters
      2. cuckoo_fail_rate.png          – insertion failure rate (Cuckoo)
      3. quotient_avg_probe_length.png – average probe length (Quotient)
      4. cuckoo_eviction_stash.png     – evictions + stash hits on same plot (Cuckoo)
      5. quotient_cluster_length.png   – cluster length histogram (Quotient)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df, run_dir = _load_results("insert_delete")
    plot_dir = _ensure_plot_dir(run_dir)
    use_mean = _has_aggregate(df)

    def col(base):
        """Return correct column name (_mean if aggregated)."""
        if use_mean and f"{base}_mean" in df.columns:
            return f"{base}_mean"
        return base

    # ============================================================
    # 1. Insert/Delete throughput
    # ============================================================
    plt.figure(figsize=(6, 4))
    for name, group in df.groupby("filter"):
        sub = group.sort_values(col("load_factor"))
        x = sub[col("load_factor")].to_numpy()
        y_ins = sub[col("insert_ops_s")].to_numpy() / 1e6
        y_del = sub[col("delete_ops_s")].to_numpy() / 1e6
        y_ins_err = (
            sub.get("insert_ops_s_std", pd.Series(np.zeros(len(sub)))).to_numpy() / 1e6
            if use_mean and "insert_ops_s_std" in sub.columns else None
        )
        y_del_err = (
            sub.get("delete_ops_s_std", pd.Series(np.zeros(len(sub)))).to_numpy() / 1e6
            if use_mean and "delete_ops_s_std" in sub.columns else None
        )
        plt.errorbar(x, y_ins, yerr=y_ins_err, fmt="o-", capsize=3, label=f"{name} insert")
        plt.errorbar(x, y_del, yerr=y_del_err, fmt="s--", capsize=3, label=f"{name} delete")
    plt.xlabel("Load Factor")
    plt.ylabel("Throughput (Million ops/s)")
    plt.title("Insert & Delete Throughput")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "insert_delete.png", dpi=150)
    plt.close()

    # Split by filter type
    cuckoo = df[df["filter"].str.contains("cuckoo", case=False)]
    qf = df[df["filter"].str.contains("quotient", case=False)]

    # ============================================================
    # 2. Cuckoo insertion failure rate
    # ============================================================
    if not cuckoo.empty and col("fail_rate") in cuckoo.columns:
        sub = cuckoo.sort_values(col("load_factor"))
        x = sub[col("load_factor")].to_numpy()
        y = sub[col("fail_rate")].to_numpy() * 100
        yerr = (
            sub.get("fail_rate_std", pd.Series(np.zeros(len(sub)))).to_numpy() * 100
            if use_mean and "fail_rate_std" in sub.columns else None
        )
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, label="Cuckoo")
        plt.xlabel("Load Factor")
        plt.ylabel("Failure Rate (%)")
        plt.title("Cuckoo Insert Failure Rate")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "cuckoo_fail_rate.png", dpi=150)
        plt.close()

    # ============================================================
    # 3. Quotient average probe length
    # ============================================================
    if not qf.empty and col("avg_probe_length") in qf.columns:
        sub = qf.sort_values(col("load_factor"))
        x = sub[col("load_factor")].to_numpy()
        y = sub[col("avg_probe_length")].to_numpy()
        yerr = (
            sub.get("avg_probe_length_std", pd.Series(np.zeros(len(sub)))).to_numpy()
            if use_mean and "avg_probe_length_std" in sub.columns else None
        )
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, label="Quotient")
        plt.xlabel("Load Factor")
        plt.ylabel("Average Probe Length (slots)")
        plt.title("Quotient Average Probe Length")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "quotient_avg_probe_length.png", dpi=150)
        plt.close()

    # ============================================================
    # 4. Cuckoo bounded evictions
    # ============================================================
    if not cuckoo.empty and col("evictions") in cuckoo.columns:
        sub = cuckoo.sort_values(col("load_factor"))
        x = sub[col("load_factor")].to_numpy()
        y = sub[col("evictions")].to_numpy()
        yerr = (
            sub.get("evictions_std", pd.Series(np.zeros(len(sub)))).to_numpy()
            if use_mean and "evictions_std" in sub.columns else None
        )
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, color="tab:blue", label="Evictions")
        plt.xlabel("Load Factor")
        plt.ylabel("Evictions per Insert Trial")
        plt.title("Cuckoo Bounded Evictions")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "cuckoo_evictions.png", dpi=150)
        plt.close()

    # ============================================================
    # 5. Cuckoo stash hits
    # ============================================================
    if not cuckoo.empty and col("stash_hits") in cuckoo.columns:
        sub = cuckoo.sort_values(col("load_factor"))
        x = sub[col("load_factor")].to_numpy()
        y = sub[col("stash_hits")].to_numpy()
        yerr = (
            sub.get("stash_hits_std", pd.Series(np.zeros(len(sub)))).to_numpy()
            if use_mean and "stash_hits_std" in sub.columns else None
        )
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, y, yerr=yerr, fmt="s--", capsize=3, color="tab:orange", label="Stash Hits")
        plt.xlabel("Load Factor")
        plt.ylabel("Stash Inserts per Trial")
        plt.title("Cuckoo Stash Hits")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "cuckoo_stash_hits.png", dpi=150)
        plt.close()


    # ============================================================
    # 6. Quotient cluster length histogram (real data; prefers summary, falls back to raw)
    # ============================================================
    # Try summary first
    hist_ok = False
    if not qf.empty and "cluster_hist" in qf.columns:
        hists = _parse_hist_series(qf["cluster_hist"])
        mean_hist = _average_hists(hists)
        if mean_hist is not None:
            bins = np.arange(len(mean_hist))
            plt.figure(figsize=(6, 4))
            plt.bar(bins, mean_hist, width=0.9, color="tab:blue", alpha=0.75)
            plt.xlabel("Cluster Length (slots)")
            plt.ylabel("Relative Frequency")
            plt.title("Quotient Cluster Length Histogram")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(plot_dir / "quotient_cluster_length.png", dpi=150)
            plt.close()
            hist_ok = True

    # Fallback: pull raw results.csv if summary lacked hist
    if not hist_ok:
        raw_csv = (run_dir / "results.csv")
        if raw_csv.exists():
            df_raw = pd.read_csv(raw_csv)
            qf_raw = df_raw[df_raw["filter"].str.contains("quotient", case=False)]
            if "cluster_hist" in qf_raw.columns:
                hists = _parse_hist_series(qf_raw["cluster_hist"])
                mean_hist = _average_hists(hists)
                if mean_hist is not None:
                    bins = np.arange(len(mean_hist))
                    plt.figure(figsize=(6, 4))
                    plt.bar(bins, mean_hist, width=0.9, color="tab:blue", alpha=0.75)
                    plt.xlabel("Cluster Length (slots)")
                    plt.ylabel("Relative Frequency")
                    plt.title("Quotient Cluster Length Histogram")
                    plt.grid(True, linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    plt.savefig(plot_dir / "quotient_cluster_length.png", dpi=150)
                    plt.close()

    print(f"[INFO] Insert/delete plots written to {plot_dir}")


# ------------------------------------------------------------
# 4. Thread Scaling
# ------------------------------------------------------------
def plot_thread_scaling():
    """
    Final, rubric-aligned thread-scaling visualization for ACS Project A3.

    Generates exactly one figure:
        thread_scaling.png  – Two subplots (Balanced | Read-mostly)
    Features:
      • One color per filter
      • Solid line + error bars (mean ± std)
      • Left: Balanced workload
      • Right: Read-mostly workload
      • Shared legend and unified styling
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Load and normalize columns
    # ------------------------------------------------------------
    df, run_dir = _load_results("thread_scaling")
    plot_dir = _ensure_plot_dir(run_dir)
    use_mean = _has_aggregate(df)

    rename_map = {"threads_mean": "threads", "qps_mean": "qps"}
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    for col in ["filter", "workload", "threads", "qps"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in thread_scaling results.")

    # Helper for safe conversion
    def arr(x):
        return np.asarray(pd.to_numeric(x, errors="coerce"), dtype=float)

    # ------------------------------------------------------------
    # Style setup
    # ------------------------------------------------------------
    color_map = {
        "bloom": "#1f77b4",     # blue
        "xor": "#ff7f0e",       # orange
        "cuckoo": "#2ca02c",    # green
        "quotient": "#d62728",  # red
    }
    workloads = ["balanced", "read_mostly"]

    # ------------------------------------------------------------
    # Create two subplots (Balanced | Read-mostly)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, wl in zip(axes, workloads):
        ax.set_title(f"{wl.replace('_', ' ').title()} Workload")
        ax.set_xlabel("Threads")
        if wl == "balanced":
            ax.set_ylabel("Throughput (Million Ops/s)")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Plot each filter
        for f in sorted(df["filter"].unique()):
            sub = df[(df["filter"] == f) & (df["workload"] == wl)]
            if sub.empty:
                continue
            sub = sub.sort_values("threads")
            x = arr(sub["threads"])
            y = arr(sub["qps"]) / 1e6
            yerr = arr(sub["qps_std"]) / 1e6 if use_mean and "qps_std" in df.columns else None

            ax.errorbar(
                x, y, yerr=yerr,
                fmt="-o", capsize=3, linewidth=2,
                color=color_map.get(f, None),
                label=f
            )

            # --- Contention point detection (robust) ---
            # Heuristic A: efficiency vs ideal linear scaling
            #   S(T) = y / y[0]; I(T) = x / x[0]; E(T) = S/I
            #   First T where E(T) < 0.8 is the contention point.
            idx = None
            if len(x) >= 3 and y[0] > 0:
                T = x.astype(float)
                S = y / y[0]
                I = T / T[0]
                E = S / I

                for i in range(1, len(E)):
                    if np.isfinite(E[i]) and E[i] < 0.8:
                        idx = i
                        break

                # Heuristic B (fallback): biggest drop in marginal slope
                if idx is None and len(y) >= 3:
                    slopes = np.diff(y) / np.diff(x)                # dQ/dT
                    rel_drop = slopes[1:] / np.maximum(slopes[:-1], 1e-12)
                    if rel_drop.size > 0:
                        j = int(np.argmin(rel_drop)) + 1            # index in y where drop occurs
                        if rel_drop[j-1] < 0.6:                     # ≥40% slope loss
                            idx = j

            if idx is not None:
                Tc = x[idx]
                Yc = y[idx]
                ax.scatter(
                    Tc, Yc,
                    color=color_map.get(f, None),
                    marker="v", s=40, zorder=5,
                    edgecolor="black", linewidths=0.6
                )

        ax.set_xlim(left=0.8, right=df["threads"].max() + 0.2)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="both", labelsize=9)

    # ------------------------------------------------------------
    # Shared legend + unified title
    # ------------------------------------------------------------
    # Shared legend (filters) + contention marker note
    handles, labels = axes[0].get_legend_handles_labels()

    # Add a dummy handle for the contention point symbol ▼
    triangle_handle = plt.Line2D(
        [], [], color="black", marker="v", linestyle="None",
        markersize=6, label="contention point"
    )
    handles.append(triangle_handle)
    labels.append("contention point")

    # Place legend centered below both subplots
    fig.legend(handles, labels, loc="lower center", ncol=5,
            fontsize="small", frameon=False, columnspacing=0.8)

    plt.suptitle("Thread Scaling (Throughput vs Threads)", fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_path = plot_dir / "thread_scaling.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[PLOT] Saved thread scaling plot → {out_path}")



# ------------------------------------------------------------
# Unified entry point
# ------------------------------------------------------------
def plot_experiment(exp_names):
    """
    Main entry invoked by experiment_manager.
    Handles list of experiment names (space, lookup, insert, threads, all).
    """
    if isinstance(exp_names, str):
        exp_names = [exp_names]

    for exp in exp_names:
        try:
            if exp == "space":
                plot_space_vs_accuracy()
            elif exp == "lookup":
                plot_lookup_latency()
            elif exp == "insert":
                plot_insert_delete()
            elif exp == "threads":
                plot_thread_scaling()
            elif exp == "all":
                plot_space_vs_accuracy()
                plot_lookup_latency()
                plot_insert_delete()
                plot_thread_scaling()
            else:
                print(f"[WARN] Unknown experiment type '{exp}'")
        except Exception as e:
            print(f"[WARN] Plot for {exp} failed: {e}")


# ------------------------------------------------------------
# CLI convenience
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_experiment(sys.argv[1:])
    else:
        print("Usage: python3 -m experiments.plot_manager <space|lookup|insert|threads|all>")
