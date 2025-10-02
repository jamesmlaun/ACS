# Memory Experiment Guide

This workspace contains seven experiment scripts under `experiments/`. Each script invokes Intel MLC (v3.11b) and/or `perf` microbenchmarks, then writes timestamped logs, CSVs, and plots. Warm-up is enabled by default; add `--no-warmup` to skip the initial pass. Trials are randomized per repetition; add `--fixed-order` to disable randomization.

## Experiments
- `zero_queue_baseline.py` — `zero_queue_baseline` — Single-in-flight MLC measurements establishing latency/bandwidth baselines.
- `pattern_granularity.py` — `pattern_granularity` — Sequential vs random access across cache-line granularities/strides.
- `read_write_mix.py` — `rw_mix` — Read/write ratio sweep; reports throughput and latency trade-offs.
- `intensity_sweep.py` — `intensity_sweep` — Arithmetic-intensity sweep to expose Little’s Law knee and bandwidth saturation.
- `working_set_sweep.py` — `working_set_sweep` — Working-set sweep to identify L1/L2/L3/DRAM transitions.
- `cache_miss_impact.py` — `cache_miss_impact` — Correlates `perf` miss events with runtime for a micro-kernel.
- `tlb_miss_impact.py` — `tlb_miss_impact` — DTLB reach and huge-page effects; includes page-size variants.

## Running Experiments

Use `manager.py` with the experiment name:

- Example: `python manager.py --exp working_set_sweep`

### Optional Flags
- Repetitions: `--reps 3` (default 3)
- Skip warm-up: `--no-warmup` (default is warm up)
- Fixed ordering: `--fixed-order` (default is random trial order)

## Output Layout

logs/:    `<experiment>_<timestamp>.log`  
results/: `<experiment>_<timestamp>.csv`  
plots/:   `<experiment>_<timestamp>/`

CSV fields include raw measurements (latency in cycles/ns, bandwidth in MB/s, miss counters) plus experiment-specific metadata; plots mirror CSV metrics with error bars where applicable.
