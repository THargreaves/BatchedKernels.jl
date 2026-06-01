#!/usr/bin/env bash
set -uo pipefail   # note: no -e
export GKSwstype=100   # GR headless: no window, savefig still works

# ======================================================================
# usage: ./run_benchmarks.sh [script|all] [force]
#   script : one of the names below, or "all" / omitted for every one.
#   force  : the literal word "force" -- passed through to each
#            run_script.jl as ARGS[1]="force", telling it to ignore its
#            JLD2 cache and re-measure. May appear in either position:
#              ./run_benchmarks.sh                 all, use cache
#              ./run_benchmarks.sh force           all, ignore cache
#              ./run_benchmarks.sh kalman          kalman only, use cache
#              ./run_benchmarks.sh kalman force    kalman only, ignore cache
# ======================================================================

# Resolve the directory this script lives in, so paths work from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
cholesky
kalman
matadd
matmul
trig_backsolve
qr_q
qr_r
sqrt_kalman
gauss_likelihood
)

# --- parse args: a "force" token in any position; the other is the
#     target script name (or "all").
TARGET=""
FORCE="0"
for a in "${1:-}" "${2:-}"; do
    case "$a" in
        force)    FORCE="1" ;;
        all|"")   ;;
        *)        TARGET="$a" ;;
    esac
done

# If a target was named, validate it and run only that script.
if [[ -n "$TARGET" ]]; then
    found=0
    for s in "${scripts[@]}"; do
        [[ "$s" == "$TARGET" ]] && found=1 && break
    done
    if [[ $found -eq 0 ]]; then
        echo "Error: unknown script '$TARGET'"
        echo "Valid options are:"
        printf '  %s\n' "${scripts[@]}"
        exit 1
    fi
    scripts=("$TARGET")
fi

# argument forwarded to each run_script.jl: "force" or nothing.
FORCE_ARG=""
[[ "$FORCE" == "1" ]] && FORCE_ARG="force"

mkdir -p "$SCRIPT_DIR/logs"

for s in "${scripts[@]}"; do
    echo "=== [$(date)] Starting $s  (force=$FORCE) ==="
    if julia --project="$SCRIPT_DIR/../." \
             "$SCRIPT_DIR/bench_$s/run_script.jl" $FORCE_ARG \
             2>&1 | tee "$SCRIPT_DIR/logs/$s.log"; then
        echo "=== [$(date)] Finished $s OK ==="
    else
        echo "=== [$(date)] $s FAILED (exit $?) ==="
    fi
done

echo "=== [$(date)] All done ==="