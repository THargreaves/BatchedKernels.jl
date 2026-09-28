#!/usr/bin/env bash
# Same operation bodies; both baselines and mixed placements use dynamic allocation.
set -euo pipefail
export SHARED_MEMORY=dynamic
export PRECISIONS="${PRECISIONS:-Float64}"
export CASES="${CASES:-32:32,32:64,32:128}"
export POLICIES="${POLICIES:-register,register_row_stage,shared_H_predicted,shared_H_predicted_row_stage}"
bash "$(dirname -- "${BASH_SOURCE[0]}")/run_precision_benchmark.sh"
