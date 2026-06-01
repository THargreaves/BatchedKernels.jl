#!/bin/bash
# ======================================================================
# run_all.sh -- full overnight pipeline, in dependency order.
#
# usage: ./run_all.sh [force]
#   force : the literal word "force" -- propagated to every step (clears
#           caches, reprofiles, re-measures). Omit to keep all caching.
#
# STEPS, in order (the order matters -- it is a dependency chain):
#
#   1. roofline baseline      profiles `ours` at nthreads=256. Produces
#                             the regs/thread that step 2 consumes.
#   2. nthreads tuning        reads the baseline profiles, writes
#                             config/nthreads_schedule.csv.
#   3. benchmarks             cross-implementation timing tables/figures.
#   4. roofline tuned         profiles at the tuned block size (needs the
#                             schedule from step 2).
#   5. comparison: sqrt_kalman block vs pad
#   6. kalman bank-conflict ablation
#   7. repeated-mul mask vs defrag
#   8. comparison: tuned vs untuned (kalman)
#
# FAILURE POLICY:
#   Steps 1-2 are the DEPENDENCY CHAIN: if either fails the run ABORTS
#   (tuning cannot proceed without baseline profiles; nothing downstream
#   can proceed without the schedule). Steps 3-8 are INDEPENDENT: a
#   failure is recorded and the run CONTINUES. A summary is printed at
#   the end so a failed step is visible without trawling the logs.
# ======================================================================
set -u   # NOT -e: independent steps must survive a failure.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # studies/
PROJECT="$HERE/."                                   # the Julia project
DIMS="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32"

# --- global force flag -------------------------------------------------
FORCE_ARG=""
[[ "${1:-}" == "force" ]] && FORCE_ARG="force"

# --- ONE-TIME OVERRIDE -------------------------------------------------
# In general the baseline roofline follows the global force flag. For
# this particular run the baseline has ALREADY been profiled and the
# expensive ncu pass should NOT be repeated, so it is pinned non-forced.
# To restore the general behaviour, set this to "$FORCE_ARG".
BASELINE_FORCE=""        # <-- one-time: baseline NOT forced. General: "$FORCE_ARG"
# ----------------------------------------------------------------------

mkdir -p "$HERE/logs"
declare -A STATUS        # step label -> exit code

# run a step, tee its output to a log, record its exit status.
# returns the step's exit code.
run_step() {
    local label="$1"; shift
    local log="$HERE/logs/run_all_${label}.log"
    echo
    echo "############################################################"
    echo "# [$label]  started $(date)"
    echo "#   $*"
    echo "############################################################"
    "$@" 2>&1 | tee "$log"
    local rc="${PIPESTATUS[0]}"
    STATUS["$label"]="$rc"
    echo "# [$label]  finished $(date)  (exit $rc)"
    return "$rc"
}

# ---- 1. roofline baseline  (DEPENDENCY CHAIN -- abort on failure) -----
run_step "1_roofline_baseline" \
    bash "$HERE/roofline/run_pipeline.sh" all "$BASELINE_FORCE" "$DIMS" baseline
if [[ "${STATUS[1_roofline_baseline]}" -ne 0 ]]; then
    echo
    echo "!!! ABORT: roofline baseline failed (exit ${STATUS[1_roofline_baseline]})."
    echo "    nthreads tuning needs the baseline profiles -- cannot continue."
    exit 1
fi

# ---- 2. nthreads tuning  (DEPENDENCY CHAIN -- abort on failure) -------
run_step "2_tune_nthreads" \
    bash "$HERE/tune_nthreads/run_tuning.sh" $FORCE_ARG
if [[ "${STATUS[2_tune_nthreads]}" -ne 0 ]]; then
    echo
    echo "!!! ABORT: nthreads tuning failed (exit ${STATUS[2_tune_nthreads]})."
    echo "    the tuned roofline needs config/nthreads_schedule.csv -- cannot continue."
    exit 1
fi

# ---- 3. benchmarks  (independent -- continue on failure) --------------
run_step "3_benchmarks" \
    bash "$HERE/benchmarks/run_benchmarks.sh" $FORCE_ARG

# ---- 4. roofline tuned  (independent of steps 5-7) --------------------
run_step "4_roofline_tuned" \
    bash "$HERE/roofline/run_pipeline.sh" all "$FORCE_ARG" "$DIMS" tuned

# ---- 5-7. comparison studies  (independent -- continue on failure) ----
run_step "5_cmp_sqrt_kalman_block_vs_pad" \
    julia --project="$PROJECT" \
    "$HERE/comparison_sqrt_kalman_block_vs_pad/run_script.jl" $FORCE_ARG

run_step "6_comparison_kalman_bank_conflict" \
    julia --project="$PROJECT" \
    "$HERE/comparison_kalman_bank_conflict/run_script.jl" $FORCE_ARG

run_step "7_comparison_repeated_mul_mask_vs_defrag" \
    julia --project="$PROJECT" \
    "$HERE/comparison_repeated_mul_mask_vs_defrag/run_script.jl" $FORCE_ARG

run_step "8_cmp_tuning" \
    julia --project="$PROJECT" \
    "$HERE/comparison_tuning/run_script.jl" $FORCE_ARG

# ---- summary ----------------------------------------------------------
echo
echo "############################################################"
echo "# run_all.sh summary  --  $(date)"
echo "############################################################"
fail=0
for label in 1_roofline_baseline 2_tune_nthreads 3_benchmarks \
             4_roofline_tuned 5_cmp_sqrt_kalman_block_vs_pad \
             6_comparison_kalman_bank_conflict 7_comparison_repeated_mul_mask_vs_defrag \
             8_cmp_tuning; do
    rc="${STATUS[$label]:-SKIPPED}"
    if [[ "$rc" == "SKIPPED" ]]; then
        printf "  %-34s SKIPPED\n" "$label"
    elif [[ "$rc" -eq 0 ]]; then
        printf "  %-34s OK\n" "$label"
    else
        printf "  %-34s FAILED (exit %s)\n" "$label" "$rc"
        fail=1
    fi
done
echo "############################################################"
[[ "$fail" -eq 0 ]] && echo "all steps OK" || echo "one or more steps FAILED -- see logs/"
exit "$fail"