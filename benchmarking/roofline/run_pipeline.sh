#!/bin/bash
set -e

# ======================================================================
# Single entry point for the roofline study, running the whole measurement
# and plotting pipeline.
#
# usage: bash run_pipeline.sh <operation | all> ["force" | ""] [dims] [profile]
#   <operation> : matmul, cholesky, qr_r, kalman, ...   or  "all"
#   force       : "force" reprofiles and re-measures even if cached
#   dims        : quoted D list, default "2 8 16"
#   profile     : "tuned" (default) or "baseline".
#                   tuned    -> nthreads = best_nthreads(op,D) from
#                               ../config/nthreads_schedule.csv.
#                   baseline -> kernels run at nthreads = 256. The naive
#                               run. This is to get the "before"
#                               data that motivates the tuning.#
# For one operation it runs, in order:
#   1. run_profiles.sh <op> ... <profile>  -> profiles `ours` implementations
#   2. analyse(op, impl, Ds; profile_dir)  -> per implementation
#   3. plot_roofline(op; profile)          -> renders the roofline SVG.
#
# ----------------------------------------------------------------------
# OUTPUT FILES (all under <op>/profile_results/<profile>/ and <op>/figs/<profile>/)
#
#   Human-readable analysis material:
#     analysis_<op>_<impl>.csv      per-implementation roofline table
#     <op>_<impl>_kernels.csv       per-kernel classification for baselines
#                                   (classifies whether a kernel is part of
#                                   one-off kernels)
#     <op>_<impl>_D<d>_kernel_bytes.csv   per-kernel DRAM breakdown
#     figs/<profile>/roofline_<op>.svg    the roofline plot
#
#   Machine-read cross-script communication:
#     <op>_<impl>_discovery.csv
#     <op>_ours_D<d>.ncu-rep / .csv / _sass.csv
#     <op>_<impl>_D<d>.csv
#
# Logs into logs/<op>_<profile>.log
# ======================================================================

ARG="${1:?usage: run_pipeline.sh <operation|all> [force] [dims] [profile]}"
FORCE="${2:-}"
DIMS="${3:-2 8 16}"
PROFILE="${4:-tuned}"

case "$PROFILE" in
    baseline|tuned) ;;
    *) echo "run_pipeline.sh: profile must be 'baseline' or 'tuned', got '$PROFILE'" >&2
       exit 1 ;;
esac

cd "$(dirname "$0")"
mkdir -p logs

# Check that the nthreads schedule exists, if not, fail early
if [[ "$PROFILE" == "tuned" && ! -f "../config/nthreads_schedule.csv" ]]; then
    echo "run_pipeline.sh: profile 'tuned' requires ../config/nthreads_schedule.csv" >&2
    echo "  -> run studies/tune_nthreads/run_tuning.sh first," >&2
    echo "     or pass 'baseline' as the 4th argument for the naive run." >&2
    exit 1
fi

# operation -> implementations to analyse, in plot order. `ours` first.
declare -A OP_IMPLS=(
    [matmul]="ours cublas magma jax"
    [cholesky]="ours cusolver magma jax"
    [qr_r]="ours cublas magma jax"
    [trig_backsolve]="ours cublas magma jax"
    [kalman]="ours magma jax"
    [sqrt_kalman]="ours magma jax"
    [gauss_likelihood]="ours jax"
)

# operations covered by "all", in run order
OPERATIONS="matmul cholesky qr_r trig_backsolve kalman sqrt_kalman gauss_likelihood"

# analysis.jl's profile_dir is a path under <op>/<profile>
PROFILE_DIR="profile_results/${PROFILE}"

run_one() {
    local op="$1"
    local impls="${OP_IMPLS[$op]:?no implementations registered for '$op' in OP_IMPLS}"
    local log="logs/${op}_${PROFILE}.log"

    {
        echo
        echo "############################################################"
        echo "# pipeline: $op   (impls: $impls)   dims: $DIMS   profile: $PROFILE"
        echo "# log:      $log"
        echo "# started:  $(date)"
        echo "############################################################"

        #  1. profile `ours`
        bash run_profiles.sh "$op" "$FORCE" "$DIMS" "$PROFILE"

        #  2. analyse every implementation and plot
        local force_flag="false"
        [[ "$FORCE" == "force" ]] && force_flag="true"

        GKSwstype=100 julia --project=../. -e "
            ENV[\"GKSwstype\"] = \"100\"
            include(\"analysis.jl\")
            include(\"plot_roofline.jl\")
            op      = \"$op\"
            impls   = split(\"$impls\")
            Ds      = [ $(echo "$DIMS" | tr ' ' ',') ]
            profile = \"$PROFILE\"
            pdir    = \"$PROFILE_DIR\"
            for impl in impls
                println(\"\n--- analyse(\", op, \", \", impl, \"; \", pdir, \") ---\")
                try
                    analyse(op, impl, Ds; profile_dir=pdir, force=$force_flag)
                catch e
                    @warn \"analyse failed for \$op / \$impl\" exception=e
                end
            end
            println(\"\n--- plot_roofline(\", op, \"; profile=\", profile, \") ---\")
            try
                plot_roofline(op; profile=profile)
            catch e
                @warn \"plot_roofline failed for \$op\" exception=e
            end
            println(\"\n--- plot_shmem_roofline(\", op, \"; profile=\", profile, \") ---\")
            try
                plot_shmem_roofline(op; profile=profile)
            catch e
                @warn \"plot_shmem_roofline failed for \$op\" exception=e
            end
        "

        echo "# finished: $(date)"
    } 2>&1 | tee "$log"
}

# Setting up the shared-memory bandwidth cache once, up front
# force does not remeasure this. Delete the csv file to re-measure
setup_shmem_bw() {
    local log="logs/_measure_shmem_bw.log"
    {
        echo "=== Setting up shared-memory bandwidth (cached unless file absent) ==="
        echo "# log:     $log"
        echo "# started: $(date)"
        GKSwstype=100 julia --project=../. -e "
            include(\"plot_roofline.jl\")
            try
                bw = shmem_bandwidth(force=false)
                println(\"shared-memory bandwidth: \", round(bw/1e12, digits=2), \" TB/s\")
            catch e
                @warn \"shmem_bandwidth measurement failed\" exception=e
            end
        "
        echo "# finished: $(date)"
    } 2>&1 | tee "$log"
}

setup_shmem_bw

if [[ "$ARG" == "all" ]]; then
    echo "=== pipeline: ALL operations ($OPERATIONS)  profile: $PROFILE ==="
    for op in $OPERATIONS; do
        run_one "$op"
    done
    echo
    echo "=== ALL operations complete ($PROFILE) ==="
else
    run_one "$ARG"
fi