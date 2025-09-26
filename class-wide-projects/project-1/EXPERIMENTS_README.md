# SIMD Performance Analysis Experiments

This project has been restructured to support two distinct experiments for analyzing SIMD performance across different cache levels.

## Experiment Overview

### Experiment 1: Cache Level Comparison
**Purpose**: Compare all kernels (SAXPY, Dot, ElemMul, Stencil) across different cache levels.

**Methodology**:
- For each kernel, select 2 N values per cache level (L1, L2, L3, DRAM)
- N values are chosen at 70% and 130% of each cache boundary
- Tests all 4 kernels at strategic points across the memory hierarchy

**Cache Boundaries** (Intel i7-11850H):
- L1: 48 KB per core
- L2: 1.25 MB per core  
- L3: 24 MB shared
- DRAM: 8x L3 size (192 MB)

### Experiment 2: Dot Locality Sweep
**Purpose**: Detailed analysis of Dot product kernel performance across cache transitions.

**Methodology**:
- Dense logarithmic sweep around each cache boundary
- 15 points around L1 boundary (30% to 200% of boundary)
- 15 points around L2 boundary (30% to 200% of boundary)
- 15 points around L3 boundary (30% to 200% of boundary)
- Focuses only on Dot product kernel for detailed locality analysis

## Usage

### Command Line Interface

```bash
# Run Experiment 1 only
python3 project_manager.py 1

# Run Experiment 2 only  
python3 project_manager.py 2

# Run both experiments sequentially
python3 project_manager.py 1 --both
```

### Example Script
```bash
# Run the example script that demonstrates all options
python3 run_experiments.py
```

## Output Structure

Each experiment creates timestamped directories:

```
logs/
├── exp1_Cache_Level_Comparison_YYYYMMDD_HHMMSS.log
└── exp2_Dot_Locality_Sweep_YYYYMMDD_HHMMSS.log

results/
├── exp1_Cache_Level_Comparison_YYYYMMDD_HHMMSS.csv
└── exp2_Dot_Locality_Sweep_YYYYMMDD_HHMMSS.csv

plots/
├── exp1_Cache_Level_Comparison_YYYYMMDD_HHMMSS/
│   ├── SAXPY_gflops.png
│   ├── SAXPY_speedup.png
│   ├── Dot_gflops.png
│   ├── Dot_speedup.png
│   ├── ElemMul_gflops.png
│   ├── ElemMul_speedup.png
│   ├── Stencil_gflops.png
│   └── Stencil_speedup.png
└── exp2_Dot_Locality_Sweep_YYYYMMDD_HHMMSS/
    ├── Dot_gflops.png
    └── Dot_speedup.png
```

## Expected N Values

### Experiment 1 N Values
The script calculates N values based on each kernel's memory requirements:

- **SAXPY/Dot**: 2 arrays → N values at 70% and 130% of each cache boundary
- **ElemMul**: 3 arrays → N values adjusted for 3-array working set
- **Stencil**: 2 arrays → N values at 70% and 130% of each cache boundary

### Experiment 2 N Values  
Dense logarithmic sweep around Dot product cache boundaries:
- L1 boundary: ~6,000 elements (30% to 200% range)
- L2 boundary: ~160,000 elements (30% to 200% range)  
- L3 boundary: ~3,000,000 elements (30% to 200% range)

## Analysis

### Experiment 1 Analysis
- Compare kernel performance across cache levels
- Identify which kernels benefit most from SIMD at different memory scales
- Understand cache hierarchy impact on different computational patterns

### Experiment 2 Analysis
- Detailed locality analysis for Dot product
- Precise identification of cache transition points
- Performance cliff analysis around cache boundaries
- SIMD effectiveness across memory hierarchy levels

## Cache Boundary Calculations

The cache boundaries are calculated as:
```
N_boundary = cache_size_bytes / (num_arrays * element_size)
```

Where:
- `cache_size_bytes`: L1 (48KB), L2 (1.25MB), L3 (24MB)
- `num_arrays`: Kernel-specific (SAXPY=2, Dot=2, ElemMul=3, Stencil=2)
- `element_size`: 4 bytes (float32)

## Hardware Specifications

**Intel Core i7-11850H**:
- 8 cores, 16 threads
- L1d: 48 KB per core
- L1i: 32 KB per core  
- L2: 1.25 MB per core
- L3: 24 MB shared
- Cache line: 64 bytes
