#!/usr/bin/env bash
# Run inside an allocated A100 job; defaults are a small hypothesis-driven set.
set -euo pipefail
export CASES="${CASES:-32:32,32:64}"
export POLICIES="${POLICIES:-register,register_row_stage,shared_H_predicted,shared_H_predicted_row_stage,shared_predicted_correction,shared_H_predicted_output}"
bash "$(dirname -- "${BASH_SOURCE[0]}")/run_precision_benchmark.sh"
