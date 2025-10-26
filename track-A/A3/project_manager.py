#!/usr/bin/env python3
# ============================================================
# project_manager.py — Universal orchestration for Project A3
# ============================================================
# Provides a unified command-line interface for:
#   • Building all binaries (bench, test_filters)
#   • Running correctness tests
#   • Running ad-hoc microbenchmarks
#   • Running full experiment sweeps (space, lookup, insert, threads)
#   • Generating plots
#
# Notes:
#   - Compatible with experiments/execution pipeline described in A3 context
#   - Uses --reps (default 3) for both benchmarks and experiment sweeps
# ============================================================

import argparse
import subprocess
import os
import datetime
import sys
import time

# ---------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------
try:
    from experiments.experiment_manager import run_experiment
except ImportError:
    run_experiment = None

try:
    from experiments.plot_manager import plot_experiment
except ImportError:
    plot_experiment = None


# ---------------------------------------------------------------------
# Utility: run shell commands with error handling
# ---------------------------------------------------------------------
def run_cmd(cmd, cwd=None):
    """Execute a shell command with optional working directory."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------
# Build all targets (bench + tests)
# ---------------------------------------------------------------------
def build_project():
    """Invoke cmake build pipeline."""
    os.makedirs("build", exist_ok=True)
    run_cmd(["cmake", "-S", ".", "-B", "build"])
    run_cmd(["cmake", "--build", "build", "--config", "Release"])
    print("[INFO] Build completed successfully.\n")


# ---------------------------------------------------------------------
# Run correctness tests for one or more filters
# ---------------------------------------------------------------------
def run_tests(filters, base_seed):
    """Execute test_filters binary for each filter and report pass/fail."""
    binary = "./build/test_filters"
    if not os.path.exists(binary):
        print("[WARN] test_filters binary not found; building first...")
        build_project()

    print("[RUN] Running correctness tests...\n")
    passed, failed = 0, 0

    for i, f in enumerate(filters):
        seed = base_seed + (i + 1)
        print(f"[TEST] Running {f} filter with seed {seed}...")
        result = subprocess.run([binary, f"--filter={f}", f"--seed={seed}"])
        if result.returncode == 0:
            passed += 1
            print(f"  ✅ {f} PASSED\n")
        else:
            failed += 1
            print(f"  ❌ {f} FAILED\n")

    print(f"[SUMMARY] {passed} passed, {failed} failed.\n")


# ---------------------------------------------------------------------
# Run ad-hoc benchmarks (manual tests)
# ---------------------------------------------------------------------
def run_benchmarks(filters, fprs, threads, workloads, reps, outdir, base_seed, warm=1):
    """
    Run direct benchmark invocations via ./build/bench.
    These are manual, non-managed runs separate from experiment sweeps.
    """
    binary = "./build/bench"
    if not os.path.exists(binary):
        print("[WARN] bench binary not found; building first...")
        build_project()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"results_{timestamp}.csv")

    for i, f in enumerate(filters):
        seed = base_seed + (i + 1)
        for fp in fprs:
            for t in threads:
                for wl in workloads:
                    cmd = [
                        binary,
                        f"--filter={f}",
                        f"--fpr={fp}",
                        f"--threads={t}",
                        f"--workload={wl}",
                        f"--reps={reps}",
                        f"--out={csv_path}",
                        f"--seed={seed}",
                        f"--warm={warm}",
                    ]
                    run_cmd(cmd)

    print(f"\n[INFO] Benchmarks complete. Results saved to: {csv_path}\n")


# ---------------------------------------------------------------------
# Experiment orchestration wrapper
# ---------------------------------------------------------------------
def run_experiment_suite(exps, reps):
    """Call experiments.experiment_manager.run_experiment."""
    if run_experiment is None:
        print("[ERROR] experiment_manager not found.")
        print("Ensure experiments/experiment_manager.py exists.")
        sys.exit(1)
    build_project()
    run_experiment(exps, reps)


# ---------------------------------------------------------------------
# Plot orchestration wrapper
# ---------------------------------------------------------------------
def run_plot_suite(exps):
    """Call experiments.plot_manager.plot_experiment."""
    if plot_experiment is None:
        print("[ERROR] plot_manager not found.")
        print("Ensure experiments/plot_manager.py exists.")
        sys.exit(1)
    plot_experiment(exps)


# ---------------------------------------------------------------------
# Main CLI interface
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project A3 Manager")

    # ---- Mode ----
    parser.add_argument(
        "--mode",
        choices=["build", "test", "run", "experiment", "plot"],
        default="run",
        help="Operation mode: build, test, run (benchmarks), experiment (managed sweeps), or plot results.",
    )

    # ---- Common flags ----
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["bloom"],
        help="Filters to test or benchmark (default: bloom).",
    )
    parser.add_argument(
        "--fprs",
        nargs="+",
        type=float,
        default=[0.01],
        help="False positive rates for benchmarking.",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1, 4, 8],
        help="Thread counts for benchmarks.",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["read_only", "read_mostly", "balanced"],
        help="Workloads to benchmark.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions per configuration (default: 3).",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Output directory for results CSVs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for hashing (default: current time).",
    )
    parser.add_argument(
        "--warm",
        type=int,
        default=1,
        help="1 = warm-up enabled (default), 0 = cold run."
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        help="Experiments to run or plot: space, lookup, insert, threads, all.",
    )

    args = parser.parse_args()

    # Determine base seed
    base_seed = args.seed if args.seed is not None else int(time.time()) & 0xFFFFFFFFFFFF
    print(f"[INFO] Using base seed: {base_seed}\n")

    # ---- Dispatch by mode ----
    if args.mode == "build":
        build_project()

    elif args.mode == "test":
        build_project()
        run_tests(args.filters, base_seed)

    elif args.mode == "run":
        build_project()
        run_benchmarks(
            args.filters, args.fprs, args.threads, args.workloads,
            args.reps, args.outdir, base_seed, args.warm
        )

    elif args.mode == "experiment":
        if not args.exp:
            print("[ERROR] Must specify --exp <space|lookup|insert|threads|all>")
            sys.exit(1)
        run_experiment_suite(args.exp, args.reps)

    elif args.mode == "plot":
        if not args.exp:
            print("[ERROR] Must specify --exp <space|lookup|insert|threads|all>")
            sys.exit(1)
        run_plot_suite(args.exp)

    else:
        print(f"[ERROR] Unknown mode: {args.mode}")
        sys.exit(1)
