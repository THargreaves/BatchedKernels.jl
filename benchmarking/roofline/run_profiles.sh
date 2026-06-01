#!/bin/bash
set -e

# usage: run_profiles.sh <operation> [force] [dims] [profile]
# Note: primary usage is to be called by run_pipeline.sh, not manually
#
# For each D, under <op>/profile_results/<profile>/, this produces:
#   <op>_ours_D<d>.ncu-rep        the profile report (the SLOW step)
#   <op>_ours_D<d>.csv            raw-page CSV   (--page raw)
#   <op>_ours_D<d>_sass.csv       source-page CSV (--page source)

OP="${1:?usage: run_profiles.sh <operation> [force] [dims] [profile]}"
FORCE="${2:-}"
WARMUPS=3
DIMS="${3:-2 8 16}"
PROFILE="${4:-baseline}"

case "$PROFILE" in
    baseline|tuned) ;;
    *) echo "run_profiles.sh: profile must be 'baseline' or 'tuned', got '$PROFILE'" >&2
       exit 1 ;;
esac

# operation -> kernel-name regex passed to ncu 
declare -A KERNEL_REGEX=(
    [matmul]="kernel_matmul"
    [cholesky]="kernel_cholesky"
    [qr_r]="kernel_qr"
    [qr_q]="kernel_qr"
    [trig_backsolve]="kernel_backward_solve"
    [kalman]="kernel_kalman"
    [sqrt_kalman]="kernel_sqrt_kalman"
    [gauss_likelihood]="kernel_gauss_likelihood"
)

REGEX="${KERNEL_REGEX[$OP]:?no kernel regex registered for operation '$OP'}"
OUTDIR="${OP}/profile_results/${PROFILE}"
mkdir -p "$OUTDIR"

# resolve the block size for a given D under the chosen profile.
#   baseline -> always 256
#   tuned    -> Schedule.best_nthreads(op, D), via config/Schedule.jl,
#                   which defaults to 256 if not tuned, which is true for singular operations
#                   except QR.
resolve_nthreads() {
    local d="$1"
    if [[ "$PROFILE" == "baseline" ]]; then
        echo 256
        return
    fi
    julia --project=../. -e "
        include(\"../config/Schedule.jl\")
        print(Schedule.best_nthreads(\"$OP\", $d))
    "
}

for D in $DIMS; do
    REP="${OUTDIR}/${OP}_ours_D${D}.ncu-rep"
    CSV="${OUTDIR}/${OP}_ours_D${D}.csv"
    SASS="${OUTDIR}/${OP}_ours_D${D}_sass.csv"

    # Use cached if all three outputs exist (delete the .ncu-rep to reprofile,
    # or pass "force")
    if [[ -f "$REP" && -f "$CSV" && -f "$SASS" && "$FORCE" != "force" ]]; then
        echo "=== $OP D=$D [$PROFILE] cached, skipping (delete $REP to reprofile) ==="
        continue
    fi

    NTHREADS="$(resolve_nthreads "$D")"

    # (re)profile only if the report is missing or force was requested
    # This runs the profile_ours.jl in each operation's subdirectory
    if [[ ! -f "$REP" || "$FORCE" == "force" ]]; then
        echo "=== $OP D=$D [$PROFILE] profiling  (nthreads=$NTHREADS) ==="
        ncu --target-processes all --set full \
            --kernel-name regex:"$REGEX" \
            --launch-skip $WARMUPS --launch-count 1 \
            --export "${OUTDIR}/${OP}_ours_D${D}" --force-overwrite \
            julia --project=../. "${OP}/profile_ours.jl" $D $WARMUPS $NTHREADS
    fi

    # cheap re-exports from the (existing) .ncu-rep -- no reprofiling
    echo "=== $OP D=$D [$PROFILE] exporting CSVs ==="
    ncu --import "$REP" --csv --page raw > "$CSV"
    ncu --import "$REP" --page source --print-source sass --csv > "$SASS"
done
