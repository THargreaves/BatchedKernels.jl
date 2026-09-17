# Kalman common-input buffer reuse experiment

Goal: test whether reducing shared-memory allocation improves **runtime**, and
whether it changes the benefit of shared intermediates relative to registers.
Occupancy is a diagnostic, not the success criterion.

## Opportunity

The four-slot Kalman workload has batched P/H and block-common A/Q/R. The current
kernel loads all three common matrices at entry and gives each a dedicated padded
buffer. A's consumers finish before Q's, and Q's before R's. The experiment loads
each at its first compute consumer, sharing one buffer across these non-overlapping
intervals. Each input is still loaded once per block, with the same loader/layout.

For Float64 D32 this saves two aligned 8,448-byte regions per block:

| Threads | Register original → reuse | Mixed original → reuse |
| --- | ---: | ---: |
| 32 | 33,792 → 16,896 | 42,240 → 25,344 |
| 64 | 42,240 → 25,344 | 59,136 → 42,240 |
| 128 | 59,104 → 42,208 | 92,864 → 75,968 |

On A100 the mixed 128-thread allocation could admit two blocks per SM instead of
one, subject to compiled register and other resource limits. The register-only
baseline also benefits from the reduced allocation, so both must be retuned.

## Scope and synchronization

This is a benchmark-only generated-expression transformation in `common_reuse.jl`;
production codegen/planning and public APIs are unchanged. It intentionally
requires dynamic shared allocation, exactly three block-common matrix inputs,
disjoint generated compute-consumer intervals, and common buffers forming the
arena tail. It rejects unrecognized consumers or overlapping lifetimes. It is
not a general-purpose shared-memory allocator.

The generated fixed offsets alias the three typed views. Before each deferred
load a block-wide barrier protects previous readers in other warps; another
barrier publishes the new input. All threads, including inactive batch-tail
threads, participate. The original initial barrier is retained conservatively.
These extra barriers and the lost overlap of initial input loads may cost more
than the allocation reduction saves. Operation bodies and register placement
are otherwise unchanged; compiler scheduling/resource allocation can still change.

## Run on A100

Inside an allocated GPU job, from the repository root:

```bash
bash benchmarking/kalman/run_common_reuse_comparison.sh
```

Defaults: Float64 D32, batch 8193, 32/64/128 threads. Each geometry includes
`register`, `register_reuse_common`, `shared_H_predicted`, and
`shared_H_predicted_reuse_common`. All four are measured in the same randomized
rounds with the existing CPU-reference and input-immutability checks. The runner
records revision, source patch, device/toolchain metadata, resources and timings.

For a shorter resource/correctness run:

```bash
MODE=resources CASES=32:128 bash benchmarking/kalman/run_common_reuse_comparison.sh
```

The underlying benchmark accepts `COMMON_REUSE=off` (unchanged default), `on`, or
`both`. Compare best-to-best runtimes across geometries, as well as the paired
same-geometry ablations. A block-count improvement without a runtime improvement
is not evidence of success. Local RTX 4090 Float64 timings cannot establish the
A100 outcome.

## Local validation (RTX 4090)

Julia 1.12.6, LLVM 18.1.7, CUDA compiler 12.9; Float64 D32, 128 threads,
batch 8193. Timings are microseconds, medians of the existing nine randomized
rounds; [raw results](common_reuse_rtx4090.csv).

| Policy | Registers | Local bytes | Shared bytes | Blocks/SM | Median µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Register | 255 | 584 | 59104 | 1 | 4094.00 |
| Register + reuse | 255 | 584 | 42208 | 2 | 3850.75 |
| Mixed | 237 | 0 | 92864 | 1 | 3829.66 |
| Mixed + reuse | 254 | 0 | 75968 | 1 | 3884.95 |

All four passed the CPU reference for every batch item and left inputs unchanged;
the five-item resource run also passed, exercising a partial block. Reuse reduces
register-only time by 5.9%, while increasing mixed time by 1.4% here. The mixed
register count increases to 254 despite unchanged mathematical operation bodies.
This is not evidence of an improved mixed-storage advantage locally. The A100
experiment matters because its larger shared capacity may admit a second mixed
block at this geometry. Neither that occupancy change nor a runtime gain should
be assumed before measuring.

Compute Sanitizer racecheck passed both reused-buffer policies at Float64 D32,
128 threads, batch five: **0 hazards, 0 errors, 0 warnings**. Reproduction:

```bash
COMMON_REUSE=on MODE=resources CASES=32:128 PRECISIONS=Float64 \
POLICIES=register,shared_H_predicted SHARED_MEMORY=dynamic \
  compute-sanitizer --tool racecheck --error-exitcode 1 \
  julia --project=. --startup-file=no benchmarking/kalman/float64_storage.jl
```
