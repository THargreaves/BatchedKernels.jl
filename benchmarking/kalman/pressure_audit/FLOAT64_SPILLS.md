# Float64 Kalman address-spill audit

The remaining local allocation in the mixed kernel was not necessarily matrix
working storage. On the RTX 4090, a small, equivalent change to cooperative
transfer indexing eliminated it entirely.

Production configuration: Float64, D=32, 128 threads, five matrices (including a
partial block), `shared_H_predicted`, column output staging, dynamic shared
allocation. Julia 1.12.6, LLVM 18.1.7, CUDA compiler 12.9, sm_89; no register caps
or device-debug compilation.

| Transfer lane indexing | Registers/thread | Local bytes/thread | Shared bytes/block |
| --- | ---: | ---: | ---: |
| `mod1(tid, 32i32)` | 255 | 152 | 92864 |
| `((tid - 1i32) & 31i32) + 1i32` | 237 | 0 | 92864 |

Both passed the CPU reference comparison and input-immutability check. The new
indexing also passed all 26 assertions in the existing GPU transfer and layout
batch-tail tests. These are resource/correctness results, not A100 timing results.

## What was spilled

The original PTX contained no explicit local array or local load/store. Native
SASS contained 38 scalar `STL` and 38 scalar `LDL` instructions, accessing 38
four-byte stack locations at offsets 0 through 148. Stores occurred early
(PC 0x2bc0–0x4040), and reloads near the final output transfer
(PC 0x69c40–0x6a4a0). Inspection of their producers and consumers identified
integer/index/address computations, rather than floating-point matrix entries.
Sequential producer tracing is not a full control-flow liveness analysis, but
it agrees with the address arithmetic surrounding the reloads.

The fixed native code contains no `STL` or `LDL` instructions and reports zero
local bytes. Masking makes the lane range explicit, allowing simpler indexing
and avoiding long-lived address temporaries. Thread indices are positive and
warps contain 32 lanes, so the replacement is equivalent for every supported
launch. It preserves Int32 arithmetic, introduces no storage-policy restriction,
and applies equally to register and mixed policies.

This also limits our earlier interpretation: 152 allocated local bytes did not
mean repeated matrix-spill traffic inside the arithmetic loops. Removing this
allocation need not produce a large speed-up. Nor does zero spilling imply low
register demand: the fixed kernel still uses 237 registers/thread. Shared-memory
capacity remains a separate occupancy constraint.

The fixed register-only policy was also checked locally: 255 registers, 584 local
bytes, 59104 shared bytes, with correctness passing. Thus this fix does not remove
all register-only local allocation. It does not establish the nature of those
remaining spills, or that A100's previously reported 192 mixed local bytes have
exactly the same origin.

## Reproduction

From the repository root, capture resources and compiler artifacts:

```bash
ARTIFACT_DIR=/tmp/kalman_float64_spills \
  julia --project=. --startup-file=no benchmarking/kalman/pressure_audit/float64_spills.jl
nvdisasm -g -gi /tmp/kalman_float64_spills/stage_9.cubin > /tmp/kalman_float64_spills/stage_9.sass
```

`THREADS` defaults to 128. `CAPTURE=false` skips disassembly artifacts. Outputs go
outside the repository by default. Capture the original revision separately to
compare compiler output; do not reuse artifacts across revisions.

On A100, rerun both policies with the same updated indexing:

```bash
POLICIES=register,shared_H_predicted \
  bash benchmarking/kalman/run_dynamic_shared_comparison.sh
```

This retains the established 32/64/128-thread cases, numerical checks, randomized
timing rounds, and revision/source recording. Only those results can establish
the effect on the thesis's A100 comparison. No new operation-body redesign is
needed for this experiment.
