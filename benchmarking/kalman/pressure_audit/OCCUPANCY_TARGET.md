# A bounded occupancy target for selective shared storage

Moving to 64 threads/block gives the selective shared-H kernel enough shared-memory
headroom for a fifth resident block. The CUDA driver confirms that reducing its
allocation to 168 registers/thread reaches that target. However, forcing that
budget introduces local memory and slows the kernel by 36%. The selected clean
placement/orientation alternatives still allocate 255 registers/thread. We have
established a feasible resource target, but have not produced a spill-free kernel
that reaches it. This does not settle the benefit of a better implementation.

No production code, compiler anchors, synchronization, or framework restrictions
were added. The existing register-heavy policy already uses shared staging;
"register-heavy" does not mean zero shared memory.

## Workload and target

RTX 4090, Float32 D32, batch8193, all five Kalman inputs batched, 64 threads/block
(two matrices/block). Other toolchain and input details match the original
[pressure audit](README.md). Using all-batched inputs removes the fixed shared
allocation for common A/Q/R inputs that constrained the earlier P/H workload.
All candidates use the existing natural trace schedule.

Device attributes report 65,536 registers, 102,400 shared bytes and 1,536 maximum
resident threads per SM; reserved shared memory is 1,024 bytes/block. The reservation
is also documented in the [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/).
The shared-H kernel reports 16,888 static shared bytes/block. Five blocks fit with
the reservation; six do not. Thus this geometry can improve from eight to ten
resident warps: theoretical occupancy 16.67% to 20.83%, a 25% increase in resident
warps, not a promised 25% throughput improvement.

The register threshold is **168**, not the 200 suggested by simply dividing the
SM register total by five blocks of 64 threads. The installed CUDA occupancy
implementation (`/usr/local/cuda/include/cuda_occupancy.h`, register-limit
calculation in `cudaOccMaxBlocksPerSMRegsLimit`) models four SM register
subpartitions and allocation in units of 256 registers/warp. At 200 registers/thread,
each subpartition holds only two warps; at 168 it holds three. `CUDA.active_blocks`
confirms four blocks at cap200 and five at cap168 for the actual compiled kernels.
These are theoretical occupancy calculations, not profiler measurements of achieved
occupancy or dynamic spill traffic.

## Selected clean candidates

The correction product was the earlier allocation trigger. To test an existing
legal alternative, H was placed in dual-access storage and the correction and final
products switched to `matmul_row`, with row output staging. A single-column H cannot
support this correction orientation; the normal assignment validator rejects it.
Dual H supports both its earlier innovation-product access and this row access,
without changing framework capabilities or inserting new conversions.

| Candidate | Single / dual slots | Registers | Shared B/block | Local B/thread | Active blocks | Median µs |
|---|---:|---:|---:|---:|---:|---:|
| Register-heavy, row output stage | 1 / 0 | 255 | 8,440 | 0 | 4 | 662.8 |
| Shared H, column output stage | 2 / 0 | 255 | 16,888 | 0 | 4 | 494.1 |
| Shared H + predicted, column stage | 2 / 0 | 255 | 16,888 | 0 | 4 | 509.7 |
| Dual H, last two products row, row stage | 1 / 1 | 255 | 16,888 | 0 | 4 | 497.8 |
| Shared H, cap200 diagnostic | 2 / 0 | 200 | 16,888 | 240 | 4 | 561.5 |

The dual alternative is valid and competitive here but does not reduce allocation.
These timings do not establish that dual storage is universally useful or redundant.
The register-heavy/shared-H comparison also changes output staging, as in the
previous selective benchmark; it is not placement alone. The H versus H+predicted
pair retains the same compute variants and output staging.

## Driver-confirmed higher occupancy, with local memory

After cap200 failed to increase occupancy, a separate randomized paired run tested
cap168 against the same uncapped shared-H kernel:

| Shared-H candidate | Registers | Local B/thread | Active blocks | Theoretical occupancy | Median µs |
|---|---:|---:|---:|---:|---:|
| Uncapped | 255 | 0 | 4 | 16.67% | 511.5 |
| Cap168 diagnostic | 168 | 344 | 5 | 20.83% | 694.1 |

The extra resident block does not offset the cost introduced by forcing the lower
budget: the capped kernel is 35.7% slower in the paired comparison. Local bytes are
the compiler-reported local frame allocation, not a measured count of memory
transactions. Capped kernels are intentionally timed as diagnostics and are not
admitted as new production candidates. Use this paired baseline for the slowdown;
the first run's 494.1µs is a different session.

**Interpretation:** shared-memory headroom is now demonstrated, and high
register allocation is the remaining resource barrier to higher occupancy in the
uncapped candidates. No tested clean implementation reaches the required register
budget. The capped result cannot answer how a spill-free 168-register implementation
would perform. Further work should focus on the previously isolated correction
fusion/code-generation issue, with 168 registers and zero local memory as a concrete
target, rather than expanding the storage search or adding scheduling barriers
without evidence.

## Validation and reproduction

Every compiled candidate passed the full CPU-reference comparison (rtol3e-4,
atol3e-5) and input-preservation checks, including the partial final block. Timings
use GPU events, nine randomized rounds of twenty launches after warmup. Exact
quartiles and resources are in [occupancy_results.csv](occupancy_results.csv).
No profiler instrumentation or concurrent GPU benchmark was used.

From the repository root, run sequentially:

```sh
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/pressure_audit/occupancy_target.jl
OPENBLAS_NUM_THREADS=1 BK_OCCUPANCY_THRESHOLD_ONLY=true julia --project=. benchmarking/kalman/pressure_audit/occupancy_target.jl
```
