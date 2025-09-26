# SIMD Experiment Guide

This workspace contains five experiment scripts under `experiments/`. Each script recompiles `kernels.cpp` into scalar (`kernels_scalar`) and SIMD (`kernels_simd`) binaries, then writes timestamped logs, CSVs, and plots. Add `--warmup` to run and discard an initial cache-warming pass.

## Experiments
- `baseline_correctness.py` — `baseline` — Samples SAXPY, Dot, ElemMul, and Stencil at cache-sized midpoints with validation and combined plots.
- `locality_sweep.py` — `locality` — Dense logarithmic sweep per kernel to expose cache transitions and cycles-per-element trends.
- `alignment_tail.py` — `alignment` — Compares aligned vs misaligned buffers across tail variants (`no_tail_16x`, `no_tail_8x`, `tail`).
- `datatype_comparison.py` — `datatype` — Evaluates float32 versus float64 performance at cache-representative sizes.
- `stride_gather.py` — `stride` — Tests stride (1/2/8/32) and gather (blocked/random) access patterns at `N = 32e6`.

Run any experiment, `project_manager.py` can be used with the shortened experiment name, e.g.:

```bash
python project_manager.py baseline --warmup
```

## Output Layout

```
logs/    <experiment>_<timestamp>_<warm|cold>.log
results/ <experiment>_<timestamp>_<warm|cold>.csv
plots/   <experiment>_<timestamp>_<warm|cold>/
```

CSV files contain scalar/SIMD runtimes, variance, derived GFLOP/s, speedup, and experiment-specific fields (stride, dtype, alignment, etc.). Plots mirror the CSV metrics with variance-based error bars.
