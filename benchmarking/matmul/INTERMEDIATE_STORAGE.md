# Matmul-only storage of computed intermediates

A matmul-only graph can demonstrate the intended **spill-free register and
occupancy reduction** from shared intermediate storage. In the selected cases it
has not demonstrated a throughput win. This separates that question from the
Kalman factorization/compiler investigation.

A subsequent [Float64/A100 experiment](A100_FLOAT64.md) extends matmul support
and tests whether shared intermediates avoid the tighter register-capacity limit.
The Float32 results below remain unchanged.

No production implementation, compiler constraints, or framework restrictions
were changed. The benchmark targets D16 and D32; this is not a new library limit.

## Retained-intermediate example

```julia
X = A * B
Y = B * A
U = A * A
V = B * B
return ((U * V) * Y) * X
```

A and B are batched inputs. X, Y, U and V are **computed matrices**, all still
needed after V is produced in this schedule. X and Y wait while U/V are computed,
then are consumed by later matmuls. Shared candidates write X, or both X/Y, directly
to single-access shared slots and read them there in subsequent computation.
Inputs remain register-resident, all matmul variants remain `matmul_row`, and
output staging remains row-oriented. The comparison changes only intermediate
residence, using the same natural schedule and block size within each pair.
Shared slots can also serve input/output staging at nonoverlapping lifetimes;
they genuinely retain intermediate values between compute operations here.

The CPU-only exhaustive topological-order audit gives a minimum whole-owner peak
of **5D line elements** for the all-register graph, including fresh output overlap.
The natural order already attains this minimum. Reordering this fixed DAG cannot
reduce that proxy further. The audit allows neither recomputation nor algebraic
reassociation and is not a physical-register lower bound. Four retained matrix
lines mean 4D Float32 values per participating lane; a fresh output, operation
scratch and compiler temporaries are additional considerations.

## D32 results

RTX 4090, Julia 1.12.6, LLVM18.1.7, CUDA compiler/runtime12.9; Float32, batch8193.
Every candidate has **zero local memory**. The reported occupancy is theoretical
CUDA occupancy, not profiler-measured achieved occupancy.

| Intermediate residence | Threads/block | Planner live-element proxy | Registers/thread | Shared B/block | Occupancy | Median µs |
|---|---:|---:|---:|---:|---:|---:|
| All registers | 64 | 160 | 255 | 8,440 | 16.67% | 392.8 |
| X shared | 64 | 128 | 218 | 8,440 | 16.67% | 400.2 |
| X/Y shared | 64 | 96 | 168 | 16,888 | 20.83% | 408.0 |
| All registers | 128 | 160 | 255 | 16,880 | 16.67% | 393.7 |
| X shared | 128 | 128 | 218 | 16,880 | 16.67% | 401.9 |
| X/Y shared | 128 | 96 | 168 | 33,776 | 16.67% | 411.0 |

At 64 threads, sharing two intermediates removes 87 allocated registers and
crosses the previously identified 168-register occupancy threshold without a cap
or spills. Nevertheless, it takes about 3.9% longer than the matched register
candidate. At 128 threads, shared memory prevents an occupancy increase and the
shared version is about 4.4% slower. Sharing only X reuses an existing staging slot,
so the total shared allocation stays unchanged, but it does not cross a register
occupancy threshold.

The final reproducer rerun confirmed all resource counts. Its D32/64 medians
were 407.8 µs (register), 400.5 µs (X shared), and 423.0 µs (X/Y shared). Several
interquartile ranges span roughly 15 µs; X-only's small timing difference changes
sign between runs. This is not a convincing speedup for X-only, and the few-percent
D32 timing differences should not be treated as precise universal penalties.
Both runs are preserved in the CSV with explicit run labels.

The D16 check shows the same distinction: registers fall 168→128 and occupancy
rises 25%→33.33%, but time increases approximately 54.5→56.3 µs at both tested
block sizes. These measurements establish a resource benefit, not a throughput
benefit. They do not prove that either placement is optimal.

## Additional controls, including unsuccessful cases

The first graph tested was:

```julia
X = A * B
Y = B * A
U = X * X
V = Y * Y
return (U * V) * (X * Y)
```

This reuses X/Y as multiply operands more heavily. It also has four live computed
values in its natural order, but a different legal schedule lowers its
whole-owner all-register peak from 5D to 4D. It is therefore a weaker example of
unavoidable simultaneous storage than the retained-intermediate graph above.

At D32/64 threads its natural-order all-register and X/Y-shared cases allocate
255 and 168 registers with zero local memory, but take 399.2 and 479.5 µs.
At 128 threads the corresponding times are 393.1 and 410.4 µs. The register-minimum
schedule control at 64 threads allocates 249 registers for register-only and 255
for shared X/Y, with medians406.1 and449.2 µs; minimizing the register-only planner
proxy does not minimize the shared candidate's proxy or physical allocation.
D16 again gives an occupancy improvement without a speed improvement. All these
controls are included in [intermediate_results.csv](intermediate_results.csv),
rather than retaining only favorable resource results.

## What this says about high register use

255 registers is not an unavoidable property of fused matmul: the same graph and
orientations compile to 168 after moving two intermediates to shared storage.
Conversely, 4D persistent values alone are not the full register budget. Fresh
results, matmul's owned-operand snapshot, unrolled arithmetic/broadcast scheduling,
addressing, and allocation constraints must also be accounted for. The planner
proxy cannot assign each of the 255 physical registers to one of those categories.

The absence of a speed win even after obtaining the desired spill-free occupancy
increase means the negative result here cannot be blamed on a failure to lower
register allocation. Added shared accesses and ordering requirements are plausible
costs, while more resident warps need not improve throughput of an already
well-pipelined instruction stream. This is an interpretation, not a profiler-based
attribution: no instruction/latency bottleneck was measured in this experiment.
It is consistent with some high register use being productive, not proof that the
compiler's allocation is optimal or that the Kalman count is necessary.

A positive intermediate-storage throughput example remains to be found. This
benchmark provides a smaller, reproducible comparison and shows why storage
selection must consider how and when an intermediate is consumed, alongside its
lifetime and resource footprint. No register cap or additional synchronization
was used to obtain these results.

## Validation and reproduction

All measured kernels passed full-batch CPU-reference comparisons (rtol3e-4,
atol3e-5) and input-preservation checks, including a partial final block. Timings
use GPU events with nine randomized rounds of twenty launches after warmup.
No candidate with local memory is admitted for these comparisons. The CSV labels each run; compare policies within the same run, graph, dimension
and order.

Run sequentially from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/matmul/intermediate_liveness.jl
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/matmul/intermediate_storage.jl
OPENBLAS_NUM_THREADS=1 MATMUL_D=16 julia --project=. benchmarking/matmul/intermediate_storage.jl
OPENBLAS_NUM_THREADS=1 MATMUL_GRAPH=reused julia --project=. benchmarking/matmul/intermediate_storage.jl
OPENBLAS_NUM_THREADS=1 MATMUL_GRAPH=reused MATMUL_ORDER=register_min julia --project=. benchmarking/matmul/intermediate_storage.jl
OPENBLAS_NUM_THREADS=1 MATMUL_GRAPH=reused MATMUL_D=16 julia --project=. benchmarking/matmul/intermediate_storage.jl
```

The liveness audit only traces phantom matrices on the CPU. Benchmark helpers are
reused from `benchmarking/kalman/hybrid_m6.jl`; the measured graphs contain no
Kalman, Cholesky, solve, or elementwise operations.
