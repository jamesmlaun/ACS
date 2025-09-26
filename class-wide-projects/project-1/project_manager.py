#!/usr/bin/env python3
import sys
import subprocess

EXPERIMENTS = {
    "baseline": "experiments/baseline_correctness.py",
    "locality": "experiments/locality_sweep.py",
    "alignment": "experiments/alignment_tail.py",
    "stride": "experiments/stride_gather.py",
    "datatype": "experiments/datatype_comparison.py",
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in EXPERIMENTS:
        print("Usage: ./project_manager.py [baseline|locality|alignment] [--warmup]")
        sys.exit(1)

    exp = sys.argv[1]
    script = EXPERIMENTS[exp]

    extra_args = []
    if "--warmup" in sys.argv:
        extra_args.append("--warmup")

    print(f"Running experiment: {exp} {'with warmup' if '--warmup' in sys.argv else ''}")
    subprocess.run(["python3", script] + extra_args, check=True)

if __name__ == "__main__":
    main()
