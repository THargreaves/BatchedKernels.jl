#!/usr/bin/env bash
# Sourced by GPU run scripts from the repository root. Records what is needed to
# reproduce a run and refuses a dirty tree unless ALLOW_DIRTY=1, in which case the
# full tracked diff and all untracked (non-ignored) files are saved with the results.
record_run_environment() {
    local out="$1" julia_bin="$2"
    if [ -n "$(git status --porcelain)" ]; then
        if [ "${ALLOW_DIRTY:-0}" != 1 ]; then
            echo "Refusing to benchmark a dirty tree; commit first or set ALLOW_DIRTY=1." >&2
            git status --short >&2
            exit 1
        fi
        git diff HEAD > "$out/source_changes.patch"
        git ls-files --others --exclude-standard -z |
            tar --null -czf "$out/untracked_files.tar.gz" -T -
    fi
    git rev-parse HEAD > "$out/revision.txt"
    git status --short > "$out/worktree_status.txt"
    cp Manifest.toml "$out/Manifest.toml"
    nvidia-smi --query-gpu=name,driver_version,memory.total,mig.mode.current \
        --format=csv > "$out/gpu.csv" 2>&1 || true
    "$julia_bin" --project=. --startup-file=no \
        -e 'using InteractiveUtils, CUDA; versioninfo(); CUDA.versioninfo()' \
        > "$out/versioninfo.txt" 2>&1
}
