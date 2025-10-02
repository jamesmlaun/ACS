"""
Shared utilities for ACS SSD experiments
"""

import datetime
import json
import os
import subprocess
import random
from pathlib import Path

# Root directories
ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TARGET_FILE = "/mnt/y/acs_testfile"

# ----------------------------
# Logging
# ----------------------------
def log_msg(msg: str, logfile: Path = None):
    """
    Print a message to stdout and also append to logfile (if provided).
    """
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")

# ----------------------------
# Environment capture
# ----------------------------
def run_cmd(cmd):
    """Run a shell command and return stdout as string."""
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except subprocess.CalledProcessError as e:
        return f"[ERROR running '{cmd}']: {e}"

def collect_environment_info():
    """Collect system + environment info for reproducibility."""
    env_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "fio_version": run_cmd("fio --version"),
        "kernel": run_cmd("uname -a"),
        "cpu_model": run_cmd("lscpu | grep 'Model name'"),
        "cpu_cores": run_cmd("lscpu | grep '^CPU(s):'"),
        "disk_info": run_cmd("lsblk -d -o NAME,MODEL,SIZE,TYPE"),
        "nvme_smart": run_cmd("sudo smartctl -a /dev/nvme0 | head -n 20"),
        "cpu_governor_note": "CPU governor locked at 99% (Turbo disabled by Windows power plan); SMT enabled",
        "data_pattern": "Incompressible (--randrepeat=0 --norandommap --zero_buffers=0)"
    }
    return env_info

def write_log_header(log_file, env_info, args):
    """Write environment + run info at top of log."""
    with open(log_file, "w") as f:
        f.write("# ACS SSD Project - Experiment Log\n")
        f.write("# ======================================\n")
        f.write(f"# Run timestamp: {env_info['timestamp']}\n")
        f.write("#\n# Environment Information\n")
        for k, v in env_info.items():
            if k == "timestamp":
                continue
            f.write(f"# {k}: {v}\n")
        f.write("#\n# Experiment Parameters\n")
        f.write(json.dumps(vars(args), indent=2))
        f.write("\n\n")

# ----------------------------
# Run directories
# ----------------------------
def prepare_run_dirs(exp_name: str, timestamp: str):
    """
    Create log, result, and figure directories for an experiment.

    Returns: (log_dir, res_dir, fig_dir, timestamp)
    """
    log_dir = LOGS / f"{exp_name}_{timestamp}"
    res_dir = RESULTS / f"{exp_name}_{timestamp}"
    fig_dir = FIGURES / f"{exp_name}_{timestamp}"

    log_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    return log_dir, res_dir, fig_dir, timestamp


# ----------------------------
# Preconditioning
# ----------------------------
def precondition_target(target_file: str, logfile: Path, size: str):
    """Sequentially write target file once to precondition SSD."""
    log_msg(f"Preconditioning SSD with sequential write ({size} total, 1M blocks)...", logfile)
    cmd = [
        "fio",
        "--name=precondition",
        f"--filename={target_file}",
        "--rw=write",
        "--bs=1M",
        f"--size={size}",
        "--iodepth=32",
        "--direct=1",
        "--numjobs=1",
        "--output-format=terse"
    ]
    subprocess.run(cmd, check=True)
    log_msg("Preconditioning complete.", logfile)

# ----------------------------
# Workload helpers
# ----------------------------
def randomize_workloads(workloads, randomize, logfile=None):
    """
    Optionally shuffle workloads if randomize=True.
    """
    if randomize:
        random.shuffle(workloads)
        log_msg(f"Trial order randomized: {workloads}", logfile)
    else:
        log_msg("Trial order fixed (default order).", logfile)
    return workloads

def fio_cmd(name, filename, rw, bs, size, runtime, iodepth, numjobs, out_json):
    """
    Construct a fio command with incompressible payload flags baked in.
    """
    return [
        "fio",
        f"--name={name}",
        f"--filename={filename}",
        f"--rw={rw}",
        f"--bs={bs}",
        f"--size={size}",
        f"--runtime={runtime}",
        f"--iodepth={iodepth}",
        f"--numjobs={numjobs}",
        "--time_based",
        "--direct=1",
        "--group_reporting",
        "--randrepeat=0",
        "--norandommap",
        "--zero_buffers=0",
        f"--output={out_json}",
        "--output-format=json"
    ]
