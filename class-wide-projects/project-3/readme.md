# SSD Experiment Guide

This workspace contains five experiment scripts under `experiments/`. Each script invokes FIO with appropriate knobs (block size, queue depth, read/write mix) and logs results into timestamped directories. All experiments can be run directly or orchestrated via `project_manager.py`. Preconditioning is enabled by default; add `--skip-precondition` to disable.

## Experiments
- `zeroqueue.py` — `zeroqueue` — Baseline latencies at QD=1 for 4 KiB random and 128 KiB sequential reads/writes; reports avg, p95, and p99.
- `block_size_sweep.py` — `block_size_sweep` — Block-size sweep from 4 KiB → 256 KiB under sequential and random access; plots IOPS/MB/s and latency, marking IOPS - MB/s crossover.
- `readwrite_mix.py` — `readwrite_mix` — Read/write ratio sweep at 4 KiB random; runs 100%R, 100%W, 70/30, and 50/50 mixes; plots throughput and latency.
- `queue_depth_sweep.py` — `queue_depth_sweep` — Parallelism sweep using `numjobs` to vary effective queue depth; plots throughput vs latency with knee point identified by Kneedle.
- `tail_latency.py` — `tail_latency` — Percentile latency distributions (p50, p95, p99, p99.9) at mid-QD and near-knee QD; plots bar chart comparison.

## Running Experiments

Use `project_manager.py` with the experiment name:

- Example: `python project_manager.py zeroqueue`

### Optional Flags
- `--skip-precondition` : Skip the SSD preconditioning step (useful for quick debugging).  
- `--precondition-size <size>` : Set the size used for preconditioning (default: 32G).  
- `--extra ...` : Pass extra arguments directly to the experiment script (e.g., `--extra --reps 5 --randomize`).  

## Output Layout

logs/:    `<experiment>_<timestamp>/run.log`  
results/: `<experiment>_<timestamp>_<raw|summary>.csv`  
plots/:   `<experiment>_<timestamp>/`  

CSV files contain per-repetition metrics (IOPS, MB/s, latencies) and aggregated summaries with mean ± standard deviation. Plots mirror the CSV metrics with error bars where applicable. Logs include environment information (CPU, SSD model, OS, FIO version) to ensure reproducibility.