"""
Shared utilities for Project 2 experiments.
Handles environment logging, command execution, and noisy-line filtering.
"""

import subprocess

# Known noisy substrings to suppress in logs
SUPPRESS_PATTERNS = [
    "Unable to modify prefetchers",
    "enabling random access"
]


def run_cmd(cmd: list[str]) -> str:
    """
    Run a shell command and return stdout as string.
    Falls back gracefully on error.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip()
    except Exception as e:
        return f"[ERROR running {cmd}] {e}"


def log_environment(log_handle, mlc_path: str):
    """
    Write environment/system info and filtering policy to a log file.
    Includes CPU info, SMT, governor, NUMA, tool versions, and suppression details.
    """
    # Suppression policy
    log_handle.write("=== Log Filtering Information ===\n")
    log_handle.write("The following patterns were suppressed from tool output:\n")
    for pat in SUPPRESS_PATTERNS:
        log_handle.write(f"- {pat}\n")
    log_handle.write("Reason: these are environment-related warnings in WSL2 that do not affect measurement.\n")
    log_handle.write("================================\n\n")

    # Environment info
    log_handle.write("=== Environment Information ===\n")

    # CPU info
    log_handle.write("CPU information (lscpu):\n")
    log_handle.write(run_cmd(["lscpu"]) + "\n\n")

    # CPU governor (may not exist in WSL2)
    log_handle.write("CPU frequency governor:\n")
    log_handle.write(run_cmd(
        ["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"]
    ) + "\n")
    log_handle.write("Note: CPU frequency scaling managed by Windows Power Options (min/max=99% to prevent turbo).\n")
    log_handle.write("Governor not directly configurable under WSL2; assumed performance mode.\n\n")

    # Tool versions
    log_handle.write("Tool versions:\n")
    log_handle.write("MLC version:\n" + run_cmd([mlc_path, "--version"]) + "\n")
    log_handle.write("Perf version:\n" + run_cmd(["perf", "--version"]) + "\n")

    log_handle.write("===============================\n\n")


def filter_and_log(output: str, suppress_patterns=SUPPRESS_PATTERNS) -> list[str]:
    """
    Filter noisy lines from MLC output and return cleaned lines.
    Suppresses known warnings like prefetcher messages.
    """
    clean_lines = []
    for line in output.splitlines():
        if any(pat in line for pat in suppress_patterns):
            continue
        clean_lines.append(line)
    return clean_lines
