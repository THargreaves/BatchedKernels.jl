# Where the 245-register allocation appears

The large allocation is not inherent to Cholesky or the triangular solves in this
workload. With H and predicted covariance in shared memory, a fused prefix through
both solves allocates 168 registers/thread. Adding the correction product
`gain_t' * H` raises allocation to 245. The final covariance product leaves it at
245. This localizes the **compilation trigger**, not the exact live-value peak:
the added product changes register allocation and instruction scheduling across
the generated kernel, including earlier code.

No production code was changed. The precise allocator mechanism behind all 245
registers remains unresolved. In particular, these experiments do not establish
that 245 is necessary, that factor scratch accounts for the gap, or that the old
broadcast-hoisting bug has recurred.

## Prefix experiment

Conditions match the previous P/H-batched audit: RTX 4090, Float32 D=32, batch 8193,
128 threads/block; A/Q/R common shared inputs. Each prefix ends in an observable
matrix output checked against a CPU reference across the entire batch, including
the partial final block. Both input sets were checked unchanged. These are
selected operation boundaries, not a Cartesian test expansion.

| Observable output | Register-heavy | H + predicted shared |
|---|---:|---:|
| `A * P` | 133 | 140 |
| predicted covariance | 156 | 156 |
| `rhs = H * predicted` | 233 | 156 |
| innovation matrix | 235 | 167 |
| Cholesky factor | 237 | 167 |
| forward solve | 237 | 168 |
| backward solve (`gain_t`) | 237 | 168 |
| correction `I - gain_t' * H` | 255 | 245 |
| final covariance | 255 | 245 |

All allocations have zero local bytes. Exact resources and errors are in
[phase_results.csv](phase_results.csv). Prefixes use natural trace order and
producer-matching output staging. Their graphs retain the declared inputs, even
when an early output does not need them, so do not interpret early-prefix policy
differences as isolated compute differences. Earlier prefixes also allow values
needed only by later operations to die. Their resource counts are not a
full-graph per-operation liveness measurement.

The complete natural-order shared kernel reproduces 245 registers and 46,460
shared bytes/block, the same endpoint as the earlier controlled schedule. The
natural-order register endpoint also allocates 255, but uses 29,564 shared bytes
instead of 46,460 under the controlled schedule. The controlled full-kernel
ablations below retain the earlier exact schedule and assignments.

The correction's subtraction is an `IAddSubWrapped` alias, folded into its
consumer; it is not a separately materialized row subtraction or an implicit
layout transpose. The product uses `matmul_col`, which mirrors `matmul_row`:
H is read from shared memory and the solved gain contributes an owned 32-element
register snapshot. A source-level 32-element snapshot alone cannot explain 245.

## What the generated code establishes

The exact **kernel-mode** PTX for prefixes 7 and 8 contains:

| Prefix | Shared loads | FMA instructions | Shuffle instructions |
|---|---:|---:|---:|
| through backward solve | 2272 | 4096 | 3632 |
| through correction | 3296 | 5120 | 3632 |

These are static instruction counts. The additional product introduces 1024 shared
loads and 1024 FMAs, with no additional shuffles. Its PTX interleaves loads and
FMAs; it does not show all 1024 loads hoisted ahead of the arithmetic. The isolated
same product, with gain in registers and H in SingleCol storage, allocates 128
registers with zero local memory and passes the CPU comparison. Therefore 245 is
an effect of its integration into this generated kernel, not the standalone
product's register requirement.

Final-SASS liveness output needs substantially more care than the initial audit
suggested. In prefix 8, a shuffle-helper CALL row displays 207 live registers,
while the preceding row displays 87 and the following row 88. The NVIDIA-generated
helper itself is only MOV/WARPSYNC/SHFL/RET, involving a handful of registers.
That 120-register jump cannot be read as 120 newly materialized shuffle values;
call-site liveness incorporates conservative interprocedural analysis.

Even excluding CALL rows is not a verified source-level liveness measurement.
The maximum displayed non-CALL count rises from 146 in prefix 7 to 180 in prefix 8,
and the latter occurs in earlier shuffle/FMA code around SASS address 0x218f0,
not simply at the newly appended shared-load product. This reinforces that the
allocation change is global. The earlier report's 236-to 208 maximum comparison
must not be used to claim an exact 28-register live-data reduction.

## Two discriminating ablations

Both ablations run in fresh Julia processes and override only the matmul method
in memory. They do not edit the library or add supported-shape restrictions.
Both use the exact earlier controlled schedule, verify full-batch CPU results,
and check original inputs unchanged.

| Controlled full kernel | Register-heavy | H + predicted shared |
|---|---:|---:|
| baseline, prior audit | 255 | 245 |
| explicitly remove square-case lane guards | 255 | 245 |
| warp fence after each output row | 255 | 251 |

The first replaces `d <= P` by `P == D || d <= P` for the owned snapshot and
output write, exploiting the caller's established `1 <= d <= D` contract. It
leaves resource counts unchanged. The second is deliberately a scheduling
**diagnostic**, not a proposed fix: in this D32 experiment each active group is a
complete warp, so it may synchronize after each output row. It raises shared
kernel allocation to 251 and slows it to 494.6µs, versus 456.8µs in the prior baseline
session. Register-heavy takes 489.7µs versus 432.8µs previously. Those cross-run
numbers are illustrative, not a randomized paired speed comparison. All four
ablation candidates have zero local bytes and unchanged 16.67% occupancy.

Neither simple guard elimination nor row fences recover the expected register
saving. There is no justification here for adding another compiler anchor or
per-row barrier to production. The remaining question is how the backend's
whole-kernel allocation, instruction scheduling, and helper-call constraints
change when the correction product is fused. A further investigation should
compare equivalent source bodies or backend allocation diagnostics at that
boundary, rather than treating 245 as an intrinsic matrix-storage requirement.

## Reproduction

Run GPU scripts sequentially from the repository root with production accessors:

```sh
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/pressure_audit/phase_prefixes.jl
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/pressure_audit/phase_product.jl
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/pressure_audit/phase_artifacts.jl
OPENBLAS_NUM_THREADS=1 BK_PHASE_ABLATION=guards julia --project=. benchmarking/kalman/pressure_audit/phase_ablation.jl
OPENBLAS_NUM_THREADS=1 BK_PHASE_ABLATION=row_fence julia --project=. benchmarking/kalman/pressure_audit/phase_ablation.jl
```

`STAGES=7,8` selects only those prefix boundaries. Artifact scripts save ignored
cubin/LLVM/PTX files under `research/split_storage_plan/pressure_audit/` by default;
`BK_PRESSURE_ARTIFACT_DIR` overrides this. Text reflection explicitly requests
`kernel=true`, matching cubin compilation. The same reflection-version caveats
as the original audit apply.
