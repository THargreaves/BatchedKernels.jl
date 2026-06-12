#!/bin/bash
set -e

# ======================================================================
# run_tuning.sh -- single entry point for the nthreads tuning pipeline.
#
# usage: ./run_tuning.sh [step] [force]
#   step  : (omitted) | predict | benchmark | decide
#   force : the literal word "force" -- ignore the benchmark cache and
#           re-measure everything. May appear in either position, e.g.
#             ./run_tuning.sh                 run all, use cache
#             ./run_tuning.sh force           run all, ignore cache
#             ./run_tuning.sh benchmark       STEP 2 only, use cache
#             ./run_tuning.sh benchmark force STEP 2 only, ignore cache
#
# Tunes block size (nthreads) for the three occupancy-bound operations
# kalman, sqrt_kalman, gauss_likelihood. QR is register-bound and D-flat
# (nthreads cannot move its limiter) and is deliberately excluded.
#
# PIPELINE  (each step reads the previous step's CSV; nothing is hidden):
#
#   STEP 1  predict_occupancy.jl   -- cheap, always runs
#       in : roofline/<op>/profile_results/<op>_ours_D<D>.csv  (registers)
#            + the GPU (device limits, queried)
#       out: results/occupancy_prediction.csv   results/device_info.csv
#
#   STEP 2  benchmark_nthreads.jl  -- the expensive step; CACHED
#       in : results/occupancy_prediction.csv   (sets the D window)
#       out: results/nthreads_benchmarks.csv
#       cache: cache/bench_<op>_D<D>.jld2  -- one file per (op,D), each
#            holding that D's 7 nthreads timings. On a normal run a
#            cached (op,D) is loaded instead of re-benchmarked; the CSV
#            is ALWAYS rebuilt from the cache so it never goes stale.
#            To re-measure a specific cell, delete its .jld2 file.
#            To re-measure everything, pass `force` (clears the cache).
#
#   STEP 3  decide_nthreads.jl     -- cheap, always runs
#       in : results/nthreads_benchmarks.csv + occupancy_prediction.csv
#       out: results/nthreads_schedule.csv  <-- the deliverable
#            results/TUNING.md              <-- auto-generated writeup
#
# results/nthreads_schedule.csv is then consumed by the roofline pipeline
# (best_nthreads(op,D)) so subsequent profiling uses the tuned block size.
# ======================================================================

cd "$(dirname "$0")"
mkdir -p results cache

# --- parse args: a "force" token in any position; the other is the step
STEP="all"
FORCE="0"
for a in "$1" "$2"; do
    case "$a" in
        force)                        FORCE="1" ;;
        predict|benchmark|decide|all) STEP="$a" ;;
        "")                           ;;
        *) echo "unknown argument: '$a'" >&2
           echo "usage: run_tuning.sh [predict|benchmark|decide] [force]" >&2
           exit 1 ;;
    esac
done

# benchmark_nthreads.jl reads TUNE_FORCE: "1" => ignore the JLD2 cache.
export TUNE_FORCE="$FORCE"

JL="julia --project=../roofline/../.."   # same project as the roofline pipeline

run_predict() {
    echo "=== STEP 1: predict_occupancy.jl ==="
    $JL predict_occupancy.jl
}
run_benchmark() {
    echo "=== STEP 2: benchmark_nthreads.jl  (force=$FORCE) ==="
    if [[ "$FORCE" == "1" ]]; then
        echo "force: clearing cache/ before benchmarking"
        rm -f cache/bench_*.jld2
    fi
    $JL benchmark_nthreads.jl
}
run_decide() {
    echo "=== STEP 3: decide_nthreads.jl ==="
    $JL decide_nthreads.jl
}

case "$STEP" in
    predict)   run_predict ;;
    benchmark) run_benchmark ;;
    decide)    run_decide ;;
    all)
        run_predict
        run_benchmark
        run_decide
        echo
        echo "=== tuning complete ==="
        echo "schedule : results/nthreads_schedule.csv"
        echo "writeup  : results/TUNING.md"
        ;;
esac
