import argparse
from experiments import zero_queue_baseline
from experiments import pattern_granularity
from experiments import read_write_mix
from experiments import intensity_sweep
from experiments import working_set_sweep
from experiments import cache_miss_impact
from experiments import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 2: Cache & Memory Profiling Manager")
    parser.add_argument("--exp", type=str, default="zero_queue_baseline",
                        help="Experiment to run (zero_queue_baseline, pattern_granularity, rw_mix)")
    parser.add_argument("--reps", type=int, default=3,
                        help="Number of repetitions (default: 3)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Disable warm-up phase (default: enabled)")
    parser.add_argument("--fixed-order", action="store_true",
                        help="Run levels/configs in fixed order instead of randomizing each repetition")

    args = parser.parse_args()

    warmup = not args.no_warmup
    randomize = not args.fixed_order

    if args.exp == "zero_queue_baseline":
        zero_queue_baseline.run(reps=args.reps, warmup=warmup, randomize=randomize)
    elif args.exp == "pattern_granularity":
        pattern_granularity.run(reps=args.reps, warmup=warmup, randomize=randomize)
    elif args.exp == "rw_mix":
        read_write_mix.run(reps=args.reps, warmup=warmup, randomize=randomize)
    elif args.exp == "intensity_sweep":
        intensity_sweep.run(reps=args.reps, warmup=warmup, randomize=randomize)
    elif args.exp == "working_set_sweep":
        working_set_sweep.run(reps=args.reps, warmup=warmup, randomize=randomize)
    elif args.exp == "cache_miss_impact":
        cache_miss_impact.run(reps=args.reps, warmup=warmup, randomize=randomize)
    else:
        print(f"[ERROR] Unknown experiment: {args.exp}")
