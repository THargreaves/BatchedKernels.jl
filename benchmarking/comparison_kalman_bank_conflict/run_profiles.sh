#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p profiles

for D in $(seq 2 32); do
    for which in orig conflict; do
        name="${which}_${D}"

        if [[ ! -f "profiles/${name}.ncu-rep" ]]; then
            echo "=== Profiling ${name} ==="
            ncu --set full --target-processes all --import-source yes \
                --kernel-name-base demangled \
                --launch-skip 2 --launch-count 1 \
                -o "profiles/${name}" -f \
                julia --project=../../../. profile.jl "$which" "$D"
        fi

        if [[ ! -f "profiles/${name}_sass.csv" ]]; then
            ncu --import "profiles/${name}.ncu-rep" --page source --print-source sass --csv \
                > "profiles/${name}_sass.csv"
        fi

        if [[ ! -f "profiles/${name}.csv" ]]; then
            ncu --import "profiles/${name}.ncu-rep" --csv --page raw \
                > "profiles/${name}.csv"
        fi
    done
done

echo "=== Done. 62 runs complete. ==="