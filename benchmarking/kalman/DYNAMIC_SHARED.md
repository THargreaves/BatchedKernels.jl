# Bounded A100 dynamic-shared comparison

This closes the static launch-path gap in the four-slot Float64 D32 Kalman study.
The covariance equation, operation bodies, operation order, input distribution,
precision, and absence of register caps are unchanged.

Inside an allocated A100 job, from the repository root:

```sh
bash benchmarking/kalman/run_dynamic_shared_comparison.sh
```

The default is Float64 D32, batch8193, 32/64/128 threads/block. At each block size it
compares register intermediates and shared H/predicted covariance, each with row
and column output staging. Both register and mixed candidates use dynamic shared
allocation. Compare the fastest mixed configuration to the fastest register
configuration over this same set. Repeat the run before interpreting a speedup.
No extra placement search or new compute algorithm is part of this experiment.

`MODE=resources` uses batch5 and omits timing. `PRECISIONS=Float32,Float64` adds the
optional Float32 control. CASES, POLICIES, BATCH, RESULT_DIR and JULIA_BIN overrides
work as in the existing runner. Results, device/compiler information, revision and
tracked source changes go under `benchmarking/kalman/precision_runs/`. Bring back
the whole generated directory. The script does not allocate a GPU job or install
dependencies; use the same university Julia/CUDA environment as before.

## Allocation and measurement contract

`fuse(...; assignment, shared_memory=:dynamic)` opts a hybrid assignment into the
new path. Static allocation remains the default; legacy automatic planning is
unchanged. The mode is part of the compilation cache key. Shared matrix, vector,
scalar-output and common-input buffers are non-overlapping, aligned views into one
dynamic arena. Their offsets and lengths are generated as constants. The arena
size must equal the planner's conservative per-region aligned byte count.

The host checks the device opt-in per-block limit before code generation. Once
compiled, it checks dynamic plus compiler-reported static bytes and sets CUDA's
maximum-dynamic-shared function attribute when required. Every launch supplies the
arena size. Direct users of `_ensure_compiled!` must likewise configure the kernel
with `_configure_dynamic_shared!` and pass the returned bytes as `shmem`.
Production view construction omits device bounds checks because sizes and launch
bytes are validated on the host; debug-accessor builds keep these checks.

Benchmark CSV `shared_bytes` now includes static plus launch-time dynamic bytes.
Occupancy and active-block queries also receive the dynamic byte count. Reading
only CUDA.memory(kernel).shared would incorrectly report this arena as free.
`RUN` records the allocation mode and device limits are logged. Oversized plans
are explicitly skipped; no feasible candidates is an error. The original static
and mixed launchers retain their defaults; set SHARED_MEMORY=static when using
those launchers if the surrounding environment exports another mode.

Dynamic allocation can change compiler scheduling and generated resources even
with identical operation bodies. The matched dynamic register controls are therefore
mandatory; do not compare only against older static timings. This extension does
not reuse dead common-input buffers or change synchronization.

Retain the earlier static register results as an additional baseline. If the new
mixed winner only beats dynamic registers but not a faster static register result,
that is not a win over the best tested register implementation. If environments
have changed, rerun the small static baseline set with the original launcher:

```sh
SHARED_MEMORY=static PRECISIONS=Float64 CASES=32:32,32:64 \
POLICIES=register,register_row_stage \
bash benchmarking/kalman/run_precision_benchmark.sh
```

## Local validation

RTX4090 Float64 D32/batch5 checks pass for column-staging register/mixed kernels
at 64 and 128 threads, and both row-staging kernels at 128 threads. Inputs remain
unchanged. The mixed arenas use 59,136 and 92,864 bytes, respectively; these exceed
the old static limit and now launch successfully. Exact observations are in
[dynamic_shared_rtx4090_resources.csv](dynamic_shared_rtx4090_resources.csv).
CUDA memcheck reports zero errors for the 128-thread column-staging mixed case,
including a partial final block. No consumer-GPU Float64 throughput claim is made.

The focused dynamic-arena tests pass in production and debug modes, including
inferred public output types, mode-separated caching, multiple region types,
alignment/padding, input preservation and oversized-arena rejection. Existing
hybrid/Float64 matmul tests and legacy scalar/vector/composite broadcast regressions
pass. A small Float32 run validates timing, dynamic-aware occupancy reporting,
CSV output and the dedicated launcher for all four selected policies.
