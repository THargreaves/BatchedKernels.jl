#!/usr/bin/env bash
# Run inside an allocated GPU job. Scheduler, account and module setup stay external.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
julia_bin="${JULIA_BIN:-julia}"
output_dir="${RESULT_DIR:-$repo_root/benchmarking/kalman/precision_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$output_dir"
git rev-parse HEAD > "$output_dir/revision.txt"
git status --short > "$output_dir/worktree_status.txt"
git diff HEAD -- src benchmarking/kalman > "$output_dir/source_changes.patch"
export OPENBLAS_NUM_THREADS=1
export MODE="${MODE:-timing}"
export RESULTS="$output_dir/results.csv"
"$julia_bin" --project="$repo_root" --startup-file=no benchmarking/kalman/float64_storage.jl 2>&1 | tee "$output_dir/run.log"
