# Giving mixed storage a fair Kalman comparison

The objective is lower complete-kernel runtime, not zero local allocation or maximum
occupancy by itself. The workload remains the four-slot P/H-batched covariance
update, with A/Q/R common across the batch. Do not alter the math, choose a weaker
register baseline, or restrict the comparison to candidates without local memory.

## Conditions for a benefit

1. A shared matrix must remove values that are live during an expensive pressure
   interval. Predicted covariance survives the innovation factorization and both
   solves; H survives until the correction product. They are plausible storage
   candidates even though the current consumers still use private working buffers.
2. The saving must persist at the actual peak. Earlier Float32 prefix experiments
   found a large allocation jump when adding `gain_t' * H`, after relatively modest
   allocation through both solves. This is a compilation trigger, not proof that
   the peak occurs exactly at that instruction. It motivates sharing a produced
   correction/output, rather than only sharing operands which are copied into
   registers again.
3. The added shared accesses must cost less than the avoided local-memory traffic
   or the gain from additional resident work. Count compiled registers, local bytes,
   shared bytes and active blocks, then measure runtime. None alone predicts speed.
4. Shared allocation must not become the new occupancy bottleneck. Dedicated common
   inputs consume shared storage once per block; per-batch slots grow with warps per
   block. Block size must therefore be considered for register and mixed candidates.
5. The placement must use compatible access orientations and avoid extra conversions
   or synchronization where possible. The candidates below keep the current compute
   variants, natural operation order and their existing synchronization contracts.
   No compiler fences, register caps, arithmetic rewrites or new operation bodies
   are introduced in this experiment.

## Small candidate set

| Candidate | Reason to include |
|---|---|
| `register` | Register intermediates, column output staging; matched baseline |
| `register_row_stage` | Earlier competitive register baseline with row staging |
| `shared_H_predicted` | Existing best local-allocation reduction; persistent H/predicted storage |
| `shared_H_predicted_row_stage` | Same persistent placement with the alternate output staging, giving mixed storage the same staging choice as registers |
| `shared_predicted_correction` | Two computed intermediates shared; removes register correction output while gain is consumed |
| `shared_H_predicted_output` | Retains the persistent-storage choice and writes the final product to shared storage |

The final-output candidate is a control for producer/output overlap. Its gains
alone would not establish a benefit from retaining intermediates. Compare it to
`shared_H_predicted` as well as the register baselines. The correction candidate
stores the product underlying the lazy `I - ...` wrapper; it does not materialize
an extra subtraction or change the equation. All shared compute slots here are
single-access, not dual-access.

## Run on the A100

Inside an allocated job, from the repository root:

```sh
bash benchmarking/kalman/run_mixed_storage_candidates.sh
```

Defaults cover both precisions, D32, 32/64 threads per block, batch8193. The existing
runner's environment overrides and output artifacts still apply. MODE=resources
performs only correctness/resource checks at batch5. The original precision-runner
policy defaults are unchanged; the new launcher selects this focused shortlist.

The planner conservatively accounts for separate shared allocations and alignment.
Candidates exceeding the device's default per-block shared limit are reported as
`SKIP,...,static_shared_budget` in run.log before compilation, and have no timing
row. The current backend uses static allocations: this check is not a statement
about the A100's full opt-in shared-memory capacity. At D32/Float64, two shared
compute slots plus three common matrices fit at 32 threads but exceed the current
path's limit at 64 threads. Extending the backend to opt-in dynamic shared memory
is a separate possible requirement for a broader block-size comparison. Do not
interpret a skipped candidate as an algorithmic failure or a slow measurement.

Within a geometry, compare matched placements to understand the effect. For the
practical winner, compare each mixed candidate against the fastest measured
register configuration across both block sizes and staging choices. Repeat runs
on an otherwise idle A100 and retain the device/compiler log. Confirm promising
results with measured local-memory traffic and shared-memory wavefronts/bank
conflicts before attributing a speedup to spilling or occupancy. This is a bounded
search, not proof of a globally optimal register or mixed implementation.

## What would need to change if this shortlist is insufficient?

The current allocator reserves separate shared arrays for A, Q and R for the whole
block, even after their last use. Additional per-batch shared slots therefore add
to that footprint. On RTX4090 the initial Float32 screen has lower residency for
mixed storage despite fewer registers: at 64 threads, register placement permits
four blocks/SM, while two-slot mixed placement permits three. All have zero local
allocation. This is a resource tradeoff, not evidence that avoiding a register
snapshot is intrinsically impossible.

The first missing capability for a broader A100 test is opt-in dynamic shared
allocation, allowing the 64-thread Float64 mixed candidates to be measured. It
must preserve alignment, byte-budget checks, launch sizing and the same choices
for the register controls. Merely increasing a numeric budget cannot make the
current static allocations work.

Reusing dead common-input buffers could reduce the shared-memory cost, but would
require real lifetime planning and block-wide synchronization before repurposing
storage previously read by several warps. It is not a free planner tweak. Likewise,
changing the active operation's register working set should be guided by compiled
pressure/local-traffic evidence, not by the assumption that every shared input
must be consumed without register temporaries. These are possible framework
changes, not features implemented by this benchmark extension. First obtain the
A100 throughput results for the supported candidates to decide whether either is
justified.

## Local screening results (RTX4090)

D32, batch8193, production accessors, Julia 1.12.6 / CUDA toolchain 12.9. All
candidates passed CPU covariance checks and input preservation. The initial screen
and a separate matched staging-control run agree closely on register and shared
column-staging timings. Exact medians/quartiles and resources are in
[mixed_storage_float32_screen.csv](mixed_storage_float32_screen.csv) and
[mixed_storage_float32_staging.csv](mixed_storage_float32_staging.csv).

At 64 threads/block:

| Policy | Registers/thread | Blocks/SM | Median µs |
|---|---:|---:|---:|
| register, column staging | 255 | 4 | 434 |
| shared H/predicted, column staging | 245 | 3 | 510–511 |
| shared H/predicted, row staging | 168 | 3 | 674 |
| shared predicted/correction | 255 | 3 | 520 |
| shared H/predicted/final output | 245 | 3 | 509 |

All have zero local allocation. Every tested 32-thread configuration is slower
than the best 64-thread register baseline. The row-staging mixed kernel shows why
lower register allocation alone is not the goal: it does not gain residency here
and runs slower. Profiling would be needed to separate instruction/scheduling
costs from shared traffic; allocation counts do not establish that mechanism.

Float64 D32/batch5 resource checks pass for the two new output-storage candidates,
the row-staging mixed candidate, and both 64-thread register baselines. See
[mixed_storage_float64_resources.csv](mixed_storage_float64_resources.csv).
At 32 threads, local bytes/thread are 504 for predicted/correction shared, 216 for
H/predicted/final-output shared, and **104 for H/predicted shared with row staging**.
The latter improves on the earlier 216-byte mixed and 712-byte row-staging register
controls, but all still allocate 255 registers/thread. This is not a Float64
throughput result. Both register baselines also pass at 64 threads; the new mixed
64-thread configurations correctly report the static-budget skip.

The final focused launcher passes shell syntax checks and completes the staging
control timing run with CSV and provenance output. No production code, operation
algorithm, synchronization contract, or framework domain restriction changed.
