#!/usr/bin/env bash
# Four candidates per geometry: both storage policies with/without common reuse.
set -euo pipefail
export COMMON_REUSE=both
export POLICIES="${POLICIES:-register,shared_H_predicted}"
bash "$(dirname -- "${BASH_SOURCE[0]}")/run_dynamic_shared_comparison.sh"
