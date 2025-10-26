# A3 Filter Experiment Guide

This workspace implements and benchmarks four **approximate membership filters**—**Bloom**, **XOR**, **Cuckoo**, and **Quotient**—under a unified C++ and Python experiment framework.  
Each experiment quantifies trade-offs in **space efficiency**, **throughput**, **latency**, and **scalability**, following the ACS Project A3 specification.

---

## Repository Structure

```
ACS-A3/     
│       
├── filters/    
│ ├── bloom.cpp / bloom.h   
│ ├── xor.cpp / xor.h   
│ ├── cuckoo.cpp / cuckoo.h     
│ ├── quotient.cpp / quotient.h     
│ └── hash_utils.h     
│   
├── benchmark/  
│ ├── bench.cpp # Unified benchmarking harness  
│ └── test_filters.cpp  
│       
├── experiments/    
│ ├── experiment_manager.py      
│ └── plot_manager.py    
│  
├── results/    
│ ├── space_vs_accuracy/    
│ ├── lookup_latency/   
│ ├── insert_delete/    
│ └── thread_scaling/   
│       
├── CMakeLists.txt  
├── Project_A3.pdf    
├── project_manager.py    
├── readme.md    
└── report_Filters.md   
```

---

## Experiments

| Experiment | Flag | Description |
|-------------|---------|-------------|
| **Space vs Accuracy** | `space` | Measures bits-per-entry (BPE) vs false positive rate (FPR) across all filters. |
| **Lookup Throughput & Latency** | `lookup` | Measures query throughput (QPS) and tail latencies (p50/p95/p99) under different negative-lookup ratios. |
| **Insert/Delete Throughput** | `insert` | Tests insertion and deletion throughput for dynamic filters (Cuckoo, Quotient) across load factors. |
| **Thread Scaling** | `threads` | Evaluates multi-threaded throughput and latency under read-mostly and balanced workloads. |

---

## Running Experiments

Use the top-level manager script:

```bash
# Run an experiment/all experiments
python3 project_manager.py --mode=experiment --exp <space|lookup|insert|threads|all>

# Build all binaries
python3 project_manager.py --mode=build

# Run the space vs accuracy experiment with 10 repetitions and warmup
python3 project_manager.py --mode=experiment --exp space --reps 10 --warm 1

# Replot all results from the latest run
python3 project_manager.py --mode=plot --exp space
```

### Optional Flags
- `--reps <n>` : Number of repetitions (default: 3)
- `--warm <0|1>` : Enable/disable warm-up pass (default: 1)
- `--seed <n>` : Specify hash seed (default: current time)

---

## Output Layout

Each experiment produces timestamped results under `results/<experiment>/<timestamp>/`:

```
results/<experiment>/<timestamp>/   
├── results.csv       # Raw per-trial data  
├── summary.csv       # Aggregated mean ± std   
├── stdout.log        # Raw execution log     
├── stderr.log        # Error log for debugging     
└── plots/            # Generated plots for report    
```

Plots mirror key CSV metrics with consistent styling and error bars.    
Summaries include experiment parameters, measured throughput, latency percentiles, and filter metadata. 

---

## Notes

- All binaries are built via CMake with `-O3 -march=native` optimizations.
- CPU pinning (`taskset`) and warm-up runs ensure stable timing.
- Results are reproducible: running `python3 project_manager.py --mode=experiment --exp all` will regenerate the complete dataset and plots.

This repository provides a modular, reproducible, and fully automated benchmarking framework for analyzing approximate membership filters under diverse workloads.