#!/usr/bin/env python3
# ============================================================
# experiment_manager.py — Orchestrates the 4 required A3 experiments
# ============================================================
# Behavior:
#   • Runs each configuration for N repetitions (default N from project_manager.py --reps)
#   • Appends ALL raw trial rows into results.csv (one per repetition)
#   • Produces summary.csv with selected mean/std metrics per configuration
#
# Summary formatting (for each experiment type):
#   - Space vs Accuracy → minimal 7-column summary
#     filter, n, target_fpr, achieved_fpr_mean, achieved_fpr_std, bpe_mean, bpe_std
#   - Other experiments still aggregate all numeric columns by default.
# ============================================================

import itertools
import os
import subprocess
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Paths and constants
# ------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
BUILD_DIR   = ROOT / "build"
BENCH_EXEC  = BUILD_DIR / "bench"
RESULTS_DIR = ROOT / "results"

FILTERS = ["bloom", "xor", "cuckoo", "quotient"]
DYNAMIC = ["cuckoo", "quotient"]

SPACE_FPRS = [0.05, 0.01, 0.001]
SPACE_N    = [1_000_000]

LOOKUP_FPRS = [0.05, 0.01, 0.001]
LOOKUP_NEG  = [0.0, 0.5, 0.9]

INSERT_LOAD = [round(0.40 + 0.05 * i, 2) for i in range(12)]
THREAD_WL   = ["read_mostly", "balanced"]
THREAD_THR  = [1, 2, 4, 8, 16]
THREAD_FPRS = [0.01]

if os.environ.get("SMOKE", "0") == "1":
    SPACE_FPRS  = [0.01]
    LOOKUP_FPRS = [0.01]
    LOOKUP_NEG  = [0.5]
    INSERT_LOAD = [0.5]
    THREAD_WL   = ["read_mostly"]
    THREAD_THR  = [1]
    print("[SMOKE] Running reduced parameter sets")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _make_run_dir(exp_name: str) -> Path:
    out_dir = RESULTS_DIR / exp_name / _ts()
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    return out_dir


def _run(cmd, run_dir: Path):
    # Extract mode and threads if `cmd` is a list
    mode = None
    threads = 1
    if isinstance(cmd, list):
        for a in cmd:
            if a.startswith("--mode="):    mode = a.split("=", 1)[1]
            elif a.startswith("--threads="):
                try: threads = int(a.split("=", 1)[1])
                except: threads = 1
        # Pin to a range only for thread-scaling; otherwise keep old behavior
        if mode == "threads" and threads > 1:
            pin = ["taskset", "-c", f"0-{max(0, threads-1)}"]
        else:
            pin = ["taskset", "-c", "0"]
        cmd = pin + cmd
    else:
        # string command fallback
        cmd = f"taskset -c 0 {cmd}"

    print(f"[RUN] {' '.join(map(str, cmd))} (CPU pin aware of --threads)")
    with open(run_dir / "stdout.log", "a") as out, open(run_dir / "stderr.log", "a") as err:
        subprocess.run(cmd, stdout=out, stderr=err, check=True)


def _ensure_bench():
    if not BENCH_EXEC.exists():
        raise FileNotFoundError("Bench binary not found. Run build first.")


def _aggregate_results(csv_path: Path, exp_name: str):
    """
    Aggregate raw results.csv into summary.csv with mean/std.
    For 'space_vs_accuracy', only keep:
      filter, n, target_fpr, achieved_fpr_mean, achieved_fpr_std, bpe_mean, bpe_std
    For other experiments, aggregate all numeric columns (legacy behavior).
    """
    if not csv_path.exists():
        print(f"[WARN] No results.csv found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARN] Empty results.csv at {csv_path}")
        return

    # Identify numeric vs categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = [c for c in df.columns if c not in numeric_cols and c not in ("rep", "seed")]

    # Force numeric-but-categorical identifiers
    for col in ["target_fpr", "threads", "load_factor", "neg_ratio", "workload"]:
        if col in df.columns and col not in group_cols:
            group_cols.append(col)

    if not group_cols:
        print(f"[WARN] No grouping columns found for {csv_path}; skipping aggregation.")
        return

    # ------------------------------------------------------------
    # Space vs Accuracy (minimal summary)
    # ------------------------------------------------------------
    if exp_name == "space_vs_accuracy":
        keep_cols = ["filter", "n", "target_fpr", "achieved_fpr", "bpe"]
        existing = [c for c in keep_cols if c in df.columns]
        df_trim = df[existing].copy()

        agg = (
            df_trim.groupby(["filter", "n", "target_fpr"], as_index=False)
            .agg(
                achieved_fpr_mean=("achieved_fpr", "mean"),
                achieved_fpr_std=("achieved_fpr", "std"),
                bpe_mean=("bpe", "mean"),
                bpe_std=("bpe", "std"),
            )
        )

        summary_path = csv_path.parent / "summary.csv"
        agg.to_csv(summary_path, index=False)
        print(f"[AGG] Wrote compact summary ({len(agg)} rows) to {summary_path}")
        return

    # ------------------------------------------------------------
    # Lookup throughput & latency (compact summary)
    # ------------------------------------------------------------
    if exp_name == "lookup_latency":
        keep_cols = [
            "filter", "neg_ratio", "qps",
            "achieved_fpr", "target_fpr",
            "p50_ns", "p95_ns", "p99_ns"
        ]
        existing = [c for c in keep_cols if c in df.columns]
        df_trim = df[existing].copy()

        agg = (
            df_trim.groupby(["filter", "neg_ratio"], as_index=False)
            .agg(
                qps_mean=("qps", "mean"),
                qps_std=("qps", "std"),
                achieved_fpr_mean=("achieved_fpr", "mean"),
                achieved_fpr_std=("achieved_fpr", "std"),
                p50_ns_mean=("p50_ns", "mean"),
                p95_ns_mean=("p95_ns", "mean"),
                p99_ns_mean=("p99_ns", "mean"),
                p50_ns_std=("p50_ns", "std"),
                p95_ns_std=("p95_ns", "std"),
                p99_ns_std=("p99_ns", "std"),
            )
        )

        summary_path = csv_path.parent / "summary.csv"
        agg.to_csv(summary_path, index=False)
        print(f"[AGG] Wrote compact lookup summary ({len(agg)} rows) to {summary_path}")
        return
    
    # ------------------------------------------------------------
    # Insert/Delete throughput (dynamic filters: Cuckoo & Quotient)
    # ------------------------------------------------------------
    if exp_name == "insert_delete":
        keep_cols = [
            "filter", "load_factor",
            "insert_ops_s", "delete_ops_s", "fail_rate",
            "avg_probe_length", "avg_cluster_length", "max_cluster_length",
            "evictions", "stash_hits", "cluster_hist"  # include hist for parsing
        ]
        existing = [c for c in keep_cols if c in df.columns]
        df_trim = df[existing].copy()

        # Numeric aggregation (concise mean/std columns)
        agg = (
            df_trim.groupby(["filter", "load_factor"], as_index=False)
            .agg(
                insert_ops_s_mean=("insert_ops_s", "mean"),
                insert_ops_s_std=("insert_ops_s", "std"),
                delete_ops_s_mean=("delete_ops_s", "mean"),
                delete_ops_s_std=("delete_ops_s", "std"),
                fail_rate_mean=("fail_rate", "mean"),
                fail_rate_std=("fail_rate", "std"),
                avg_probe_length_mean=("avg_probe_length", "mean"),
                avg_probe_length_std=("avg_probe_length", "std"),
                avg_cluster_length_mean=("avg_cluster_length", "mean"),
                avg_cluster_length_std=("avg_cluster_length", "std"),
                max_cluster_length_mean=("max_cluster_length", "mean"),
                max_cluster_length_std=("max_cluster_length", "std"),
                evictions_mean=("evictions", "mean"),
                evictions_std=("evictions", "std"),
                stash_hits_mean=("stash_hits", "mean"),
                stash_hits_std=("stash_hits", "std"),
            )
        )

        # ---- Aggregate histogram (compact serialized, averaged & normalized) ----
        # We parse quoted "a,b,c,..." strings per row, average within each (filter,load_factor) group,
        # then normalize to sum to 1 and re-serialize as comma-separated floats.
        if "cluster_hist" in df_trim.columns:
            def _parse_hist_cell(s):
                if pd.isna(s):
                    return None
                raw = str(s).strip().strip('"')
                if not raw:
                    return None
                vals = [v.strip() for v in raw.split(",") if v.strip() != ""]
                try:
                    arr = np.array([float(v) for v in vals], dtype=float)
                    return arr
                except Exception:
                    return None

            hist_rows = []
            for (filt, lf), g in df_trim.groupby(["filter", "load_factor"]):
                parsed = [h for h in ( _parse_hist_cell(x) for x in g["cluster_hist"].tolist() ) if h is not None]
                if not parsed:
                    continue
                max_len = max(len(h) for h in parsed)
                aligned = [np.pad(h, (0, max_len - len(h)), constant_values=0.0) for h in parsed]
                mean_hist = np.mean(np.stack(aligned, axis=0), axis=0)
                total = float(mean_hist.sum())
                if total > 0:
                    mean_hist = mean_hist / total
                hist_str = ",".join(f"{x:.6f}" for x in mean_hist.tolist())
                hist_rows.append({"filter": filt, "load_factor": lf, "cluster_hist": hist_str})

            if hist_rows:
                hist_df = pd.DataFrame(hist_rows)
                # Left-join to preserve all numeric means/stds
                agg = agg.merge(hist_df, on=["filter", "load_factor"], how="left")

        # write summary
        summary_path = csv_path.parent / "summary.csv"
        agg.to_csv(summary_path, index=False)
        print(f"[AGG] Wrote compact insert/delete summary ({len(agg)} rows) to {summary_path}")
        return

    # ------------------------------------------------------------
    # Default aggregation for other experiment types
    # ------------------------------------------------------------
    agg = df.groupby(group_cols, as_index=False).agg({c: ["mean", "std"] for c in numeric_cols})
    agg.columns = ["_".join(col).rstrip("_") for col in agg.columns.values]
    summary_path = csv_path.parent / "summary.csv"
    agg.to_csv(summary_path, index=False)
    print(f"[AGG] Wrote summary statistics ({exp_name}) grouped by {group_cols} → {summary_path}")


# ------------------------------------------------------------
# 1) Space vs Accuracy
# ------------------------------------------------------------
def experiment_space_vs_accuracy(reps: int, warm: int):
    _ensure_bench()
    run_dir  = _make_run_dir("space_vs_accuracy")
    csv_path = run_dir / "results.csv"

    FIXED_N = SPACE_N[0]
    for filt, fpr in itertools.product(FILTERS, SPACE_FPRS):
        for r in range(reps):
            cmd = [
                str(BENCH_EXEC),
                "--mode=space",
                f"--filter={filt}",
                f"--n={FIXED_N}",
                f"--fpr={fpr}",
                f"--seed={r}",
                f"--out={csv_path}",
                f"--warm={warm}",
            ]
            _run(cmd, run_dir)

    _aggregate_results(csv_path, "space_vs_accuracy")
    print(f"[INFO] ✅ Space vs accuracy results stored in {run_dir}")


# ------------------------------------------------------------
# 2) Lookup throughput & latency
# ------------------------------------------------------------
def experiment_lookup_latency(reps: int, warm: int):
    _ensure_bench()
    run_dir  = _make_run_dir("lookup_latency")
    csv_path = run_dir / "results.csv"

    FIXED_FPR = 0.01
    for filt in FILTERS:
        for r in range(reps):
            for neg in LOOKUP_NEG:
                cmd = [
                    str(BENCH_EXEC),
                    "--mode=lookup",
                    f"--filter={filt}",
                    f"--fpr={FIXED_FPR}",
                    f"--neg_ratio={neg}",
                    "--threads=1",
                    "--reps=1",
                    f"--seed={r}",
                    f"--out={csv_path}",
                    f"--warm={warm}",
                ]
                _run(cmd, run_dir)

    _aggregate_results(csv_path, "lookup_latency")
    print(f"[INFO] ✅ Lookup latency results stored in {run_dir}")


# ------------------------------------------------------------
# 3) Insert/Delete throughput
# ------------------------------------------------------------
def experiment_insert_delete(reps: int, warm: int):
    _ensure_bench()
    run_dir  = _make_run_dir("insert_delete")
    csv_path = run_dir / "results.csv"

    for filt, lf in itertools.product(DYNAMIC, INSERT_LOAD):
        for r in range(reps):
            cmd = [
                str(BENCH_EXEC),
                "--mode=insert",
                f"--filter={filt}",
                f"--load_factor={lf}",
                "--n=1000000",
                "--fpr=0.01",
                f"--seed={r}",
                f"--out={csv_path}",
                f"--warm={warm}",
            ]
            _run(cmd, run_dir)

    _aggregate_results(csv_path, "insert_delete")
    print(f"[INFO] ✅ Insert/Delete results stored in {run_dir}")


# ------------------------------------------------------------
# 4) Thread scaling
# ------------------------------------------------------------
def experiment_thread_scaling(reps: int, warm: int):
    _ensure_bench()
    run_dir  = _make_run_dir("thread_scaling")
    csv_path = run_dir / "results.csv"

    for filt, wl, th, fpr in itertools.product(FILTERS, THREAD_WL, THREAD_THR, THREAD_FPRS):
        for r in range(reps):
            cmd = [
                str(BENCH_EXEC),
                "--mode=threads",
                f"--filter={filt}",
                f"--workload={wl}",
                f"--threads={th}",
                f"--fpr={fpr}",
                "--n=1000000",
                "--reps=1",
                f"--seed={r}",
                f"--out={csv_path}",
                f"--warm={warm}",
            ]
            _run(cmd, run_dir)

    _aggregate_results(csv_path, "thread_scaling")
    print(f"[INFO] ✅ Thread scaling results stored in {run_dir}")


# ------------------------------------------------------------
# Entry point for project_manager.py
# ------------------------------------------------------------
def run_experiment(exp_names, reps: int = 3, warm = 1):
    for exp in exp_names:
        print(f"\n===== Running Experiment: {exp} ({reps}×) ({'warm' if warm else 'cold'}) =====\n")
        if exp == "space":
            experiment_space_vs_accuracy(reps, warm)
        elif exp == "lookup":
            experiment_lookup_latency(reps, warm)
        elif exp == "insert":
            experiment_insert_delete(reps, warm)
        elif exp == "threads":
            experiment_thread_scaling(reps, warm)
        elif exp == "all":
            experiment_space_vs_accuracy(reps, warm)
            experiment_lookup_latency(reps, warm)
            experiment_insert_delete(reps, warm)
            experiment_thread_scaling(reps, warm)
        else:
            print(f"[WARN] Unknown experiment '{exp}', skipping.")

    print("\n✅ All experiments completed successfully.\n")

    try:
        from experiments.plot_manager import plot_experiment
        plot_experiment(exp_names)
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")
