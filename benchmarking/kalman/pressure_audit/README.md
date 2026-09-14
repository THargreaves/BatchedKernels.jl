# Kalman register-pressure audit

The current hybrid kernels do not yet demonstrate the intended occupancy benefit
from moving two long-lived matrices into shared memory. A controlled P/H-batched
comparison reduces the planner's matrix-element peak by 64 but reduces compiled
registers by only 10. Capping existing kernels near the legacy register count
introduces local memory and slows them down. This is evidence about these generated
kernels, not a lower bound on an improved implementation.

Follow-up: [phase isolation and compiler diagnostics](PHASE_ORIGIN.md) localize
the allocation jump to adding the correction product and correct the initial
disassembler-liveness interpretation below. The exact allocator cause remains open.

No production implementation, supported shapes, or resource admission rules were
changed for this investigation.

## Conditions and validation

RTX 4090 (sm_89), Julia 1.12.6, LLVM 18.1.7, CUDA compiler/runtime 12.9,
CUDACore 6.1.1, GPUCompiler 1.13.3, driver 560.35.03. Float32, D=32, batch=8193.
Each candidate was checked against the full CPU reference (rtol=3e-4,
atol=3e-5), including the partial final block. Inputs were checked unchanged.
Times are GPU event medians over nine randomized rounds of twenty launches after
warmup; quartiles and exact resources are in [results.csv](results.csv).
These are single-session measurements, not cross-device results.

## 1. Can a cap recover the legacy register count cheaply?

All five inputs are batched. These are the previous selective finalists: the
register-heavy kernel uses row output staging, the shared-H kernel uses column
output staging. Each cap is compared against its own uncapped kernel. Legacy uses
32 threads/block; the two hybrid kernels use 128. Consequently, this is not a
controlled comparison of placement alone.

| Kernel | Register cap | Actual registers | Local bytes/thread | Occupancy | Time (µs) |
|---|---:|---:|---:|---:|---:|
| Legacy | None | 167 | 0 | 8.33% | 864.0 |
| Register-heavy | None | 255 | 0 | 16.67% | 664.2 |
| Register-heavy | 192 | 192 | 256 | 16.67% | 837.5 |
| Register-heavy | 168 | 168 | 368 | 25.00% | 816.3 |
| Register-heavy | 128 | 128 | 616 | 33.33% | 1000.9 |
| Shared H | None | 255 | 0 | 16.67% | 488.4 |
| Shared H | 192 | 192 | 264 | 16.67% | 583.8 |
| Shared H | 168 | 168 | 344 | 16.67% | 786.4 |
| Shared H | 128 | 128 | 536 | 16.67% | 859.2 |

The 168 cap costs approximately 23% for register-heavy and 61% for shared H.
Higher theoretical occupancy does not compensate in the register-heavy case;
the shared-H case remains limited by its shared-memory allocation (33,776 B/block).
Local bytes are the allocated local-memory frame, not dynamic spill traffic.
Unlike the earlier benchmark admission policy, **these diagnostic runs deliberately
time kernels with local memory**. They are not newly admitted production candidates.

## 2. Does directly sharing two matrices reduce pressure?

For the workload closer to the original four-slot example, P and H are batched;
A, Q, and R are common shared inputs. Both hybrid candidates use 128 threads/block,
the same compute variants, column output staging, and the same valid schedule.
Only H/predicted-covariance residence and H's matching load staging change.

| Kernel | Planner live-element proxy | Compiled registers/thread | Shared B/block | Local B/thread | Occupancy | Time (µs) |
|---|---:|---:|---:|---:|---:|---:|
| Register-heavy | 160 | 255 | 46,460 | 0 | 16.67% | 432.8 |
| Shared H + predicted | 96 | 245 | 46,460 | 0 | 16.67% | 456.8 |

Both plans use two single-access slots; shared operands can reuse slots that were
already needed for staging. The shared byte totals include common shared inputs,
not only per-matrix slots. The shared candidate's interquartile timing range was
440.7–457.8 µs versus 432.6–450.3 µs for register-heavy, so do not overinterpret the
roughly 6% median difference. The unchanged occupancy and small register reduction
are the stronger findings. Legacy at 32 threads/block used 158 registers, 29,564 B
shared memory and 6.25% occupancy, taking 1060.9 µs on this workload.

### Relating the 4D/2D intuition to the planner

The planner counts whole register-resident matrix owners, including a fresh output
at the same operation as its inputs. It excludes operation scratch, addresses,
predicates, compiler temporaries, and scalar-level last uses.

The exact schedule audit under that model gives:

| Workload | Register-heavy minimum | H + predicted shared minimum |
|---|---:|---:|
| All five inputs batched | 6D | 4D |
| P/H batched, A/Q/R common | 5D | 3D |

For P/H batched, the innovation addition has H, predicted covariance, rhs,
innovation product, and fresh sum simultaneously live in this conservative model.
Sharing H and predicted removes two. With all inputs batched, R adds another.
The fresh sum accounts for the extra D relative to the intuitive 4D/2D model.

These are **not physical-register lower bounds**: the compiler can reuse dead
input-element registers for output elements even though whole matrix owners are
distinct. Conversely, an implementation may need considerably more than this proxy.
The controlled test demonstrates that reducing the proxy by 2D does not guarantee
reducing the actual peak by 2D.

## 3. Final assembly evidence and remaining attribution limit

The initial disassembler live-GPR maxima are recorded below for reproducibility.
Follow-up inspection found sharp count jumps at helper calls; these maxima must
not be interpreted as measured scalar-data liveness:

| Kernel | Allocated registers | Maximum displayed live GPRs |
|---|---:|---:|
| All-batched legacy | 167 | 146 |
| All-batched register-heavy | 255 | 253 |
| All-batched shared H | 255 | 245 |
| P/H register-heavy | 255 | 236 |
| P/H shared H + predicted | 245 | 208 |

These counts use NVIDIA's [`nvdisasm` register-liveness output](https://docs.nvidia.com/cuda/cuda-binary-utilities/#register-life-range-information).
They are a disassembler analysis, not interchangeable with the final allocation.
In the correction prefix, one small shuffle-helper call raises the displayed count
from 87 to 207, followed by 88 on return; the helper itself uses seven registers.
This exposes an interprocedural-analysis complication. The 236→208 maxima above
do not establish a 28-register reduction in actual scalar-data liveness. Allocation
constraints and control flow need separate analysis; the difference between the
columns is not a count of wasted registers.

In the P/H register kernel, 1,938 of the 2,413 static instruction rows with at least
220 displayed live GPRs are shuffles or calls to the generated shuffle helper.
The call-row complication means this histogram cannot locate physical pressure
in communication or establish what the live values represent. It also does not
measure execution frequency. In particular, it does **not** prove that the old
broadcast-hoisting issue recurred.

The new factorization bodies also carry private D-wide working state, and matmul
snapshots an operand line. Such source-level scratch can overlap or be coalesced
with inputs/outputs; simply adding every declared vector size is not a reliable
physical-register prediction. The legacy bodies use shared operands and different
operation implementations, so their lower count is a useful target, not proof
that placement alone can reproduce it.

The maximum-live P/H register instruction maps to embedded PTX around unrolled
shared loads, address arithmetic and additions. That PTX has no Julia source
locations, and the optimized mapping is not one-to-one. The current capture
therefore does not provide a verified Julia source map at the peak. Accordingly, this audit does not name a specific operation as the root cause.
The next discriminating experiment is to inspect prefixes or substitute one stage
of this same schedule, keeping the output observable, then compare generated
liveness and full-kernel throughput. Test scheduling constraints only after finding
the dependency transformation responsible. Prefix resource counts alone are not
full-kernel predictions because cutting the graph changes its live values.

**Decision:** retain the storage framework and the selective candidates, but do not
claim the intended 2D register saving or an occupancy benefit yet. Keep register
caps as diagnostics. No new compiler anchor or framework restriction is justified
by the evidence collected here.

## Reproduction

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/pressure_audit/kalman_liveness.jl
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/pressure_audit/caps.jl
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/pressure_audit/ph.jl
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/pressure_audit/artifacts.jl
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/pressure_audit/ph_artifacts.jl
```

Run GPU benchmarks sequentially on an otherwise idle GPU. The liveness helper is
CPU-only. The initial standalone LLVM/PTX dumps used device-function mode; the dumper now
passes `kernel=true` to match the binary captures, which already used kernel mode.
The measured resources and timings are unaffected. Artifact capture uses the
installed CUDA.jl reflection internals; it is
research tooling tied to the versions above. Generated binaries/IR are ignored.

For each generated cubin, inspect liveness with:

```sh
nvdisasm --print-code --separate-functions --print-life-ranges --life-range-mode count KERNEL.cubin
```
