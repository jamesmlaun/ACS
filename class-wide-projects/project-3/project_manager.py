#!/usr/bin/env python3
"""
ACS SSD Project - Project Manager

This script orchestrates all SSD experiments:
- Runs experiments in experiments/ directory
- Collects logs and results
- Generates plots
- Logs environment information (fio version, CPU, SSD, OS) for reproducibility
- Preconditions SSD before first experiment (configurable or skipped)
"""

import sys
import subprocess
import datetime
import argparse
from pathlib import Path

# Import shared utilities
from experiments.utils import (
    ROOT, LOGS, TARGET_FILE,
    collect_environment_info,
    write_log_header,
    precondition_target,
    log_msg
)

# ----------------------------
# Run Experiment
# ----------------------------
def run_experiment(script, args, env_info, precondition_done):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create log dir + logfile
    log_dir = LOGS / f"{script.stem}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    # Write environment info header once
    write_log_header(log_file, env_info, args)

    # Precondition SSD only once (before the first experiment) unless skipped
    if not precondition_done:
        if args.skip_precondition:
            log_msg("Skipping SSD preconditioning step (per --skip-precondition).", log_file)
        else:
            log_msg("Preconditioning SSD (manager-controlled)…", log_file)
            precondition_target(TARGET_FILE, log_file, args.precondition_size)
            precondition_done = True

    # Build experiment command
    cmd = [sys.executable, str(script),
           "--timestamp", ts,
           "--logfile", str(log_file)]
    if args.extra:
        cmd += args.extra

    # Run experiment (stdout shown live + logged)
    subprocess.run(cmd, check=False)

    log_msg(f"Experiment {script.stem} finished. Unified log: {log_file}")
    return precondition_done

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ACS SSD Project Manager - orchestrates experiments"
    )
    parser.add_argument(
        "experiment",
        choices=[p.stem for p in (ROOT / "experiments").glob("*.py")],
        help="Experiment script to run"
    )
    parser.add_argument(
        "--extra", nargs=argparse.REMAINDER,
        help="Extra arguments passed to experiment script"
    )
    parser.add_argument(
        "--skip-precondition", action="store_true",
        help="Skip the preconditioning step (useful for debugging)"
    )
    parser.add_argument(
        "--precondition-size", default="32G",
        help="Size to use for preconditioning (default: 32G)"
    )
    args = parser.parse_args()

    # Collect environment info
    env_info = collect_environment_info()

    # Locate experiment script
    script = ROOT / "experiments" / f"{args.experiment}.py"
    if not script.exists():
        print(f"[ERROR] Experiment {script} not found")
        sys.exit(1)

    # Run experiment
    precondition_done = False
    precondition_done = run_experiment(script, args, env_info, precondition_done)

if __name__ == "__main__":
    main()
