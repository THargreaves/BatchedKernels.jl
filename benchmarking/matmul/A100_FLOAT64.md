# Float64 intermediate storage on A100

This experiment asks whether retaining computed intermediates in shared memory
avoids register-capacity pressure in Float64, and whether that improves throughput
on an A100. It uses the same seven-matmul graph as INTERMEDIATE_STORAGE.md:

```julia
X = A * B
Y = B * A
U = A * A
V = B * B
return ((U * V) * Y) * X
```

Compare all-register intermediates, X in single-access shared storage, and X/Y in
single-access shared storage. Inputs remain register-resident; operation variants,
orientations and natural schedule match within each scenario. No register caps or
new synchronization are introduced. Both precisions use the same underlying input
data, rounded to the chosen type.

## Implementation scope

Named hybrid matmul variants now accept matching Float32 or Float64 operands.
Float64 elementwise, Cholesky and solve variants are not enabled by this change.
The existing generic matmul/device-accessor bodies are reused; no runtime type
switch is introduced in device code. Planner byte accounting uses the selected
scalar type. `peak_register_elements` remains a logical element proxy; the runner
also reports its 32-bit-word equivalent, which still excludes scratch/allocator
costs and is not a physical-register prediction.

The benchmark intentionally retains candidates with compiled local memory. In
Float64, comparing that case with explicit shared storage is the experiment;
`local_memory` and `zero_local` are recorded separately. This does not change the
admission policy of earlier Float32 benchmarks. Reported local bytes are the local
frame allocation, not measured dynamic spill traffic.

Scenarios are D16/64 threads (smaller-matrix control), D32/32 threads and D32/64
threads. These keep two Float64 shared slots below the existing static shared-memory
per-block limit in the current launch path. The test does not add a framework
shape/thread restriction or enable larger static allocations.

## Running on the HPC

On a login/setup node, clone or update this branch and install its environment:

```sh
julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
```

Use the site's supported Julia/CUDA setup; Julia 1.12.6 matches the local validation.
The launcher deliberately leaves scheduler partition, account, allocation and
module setup to the university's usual workflow. It does not submit a job or
connect to the HPC. Once an A100 is allocated, from the repository root run:

```sh
bash benchmarking/matmul/run_precision_benchmark.sh
```

The launcher defaults to timing mode and batch8193. `JULIA_BIN` selects a Julia
executable if it is not on PATH. `RESULT_DIR` selects the output directory;
otherwise a timestamped directory is created in `benchmarking/matmul/precision_runs`.
It records the source revision, tracked source changes, worktree status, CUDA/Julia
versions, SM count, device resource limits and result CSV. Package installation is
not attempted inside the GPU job. Keep the allocated GPU free of concurrent
benchmarks, and note whether it is a full A100 or a MIG partition when sharing results.

For a quick correctness/resource run before timing:

```sh
MODE=resources bash benchmarking/matmul/run_precision_benchmark.sh
```

That mode defaults to batch5, exercising partial blocks. It records no throughput
numbers. Individual scenarios can be selected, for example:

```sh
CASES=32:32,32:64 PRECISIONS=Float64 bash benchmarking/matmul/run_precision_benchmark.sh
```

For the primary comparison, retain both precisions and all default scenarios.
`BATCH` overrides batch size. The runner validates each batch member against a CPU
reference with Float64 rtol1e-11/atol1e-12 (Float32 rtol3e-4/atol3e-5), verifies
inputs unchanged, and times nine randomized rounds of twenty launches after warmup.
Times cover the complete fused kernel, including global input/output transfers.

## Interpreting results

Compare policies within each precision, shape and block size. First check whether
shared intermediates eliminate or reduce local memory. Then compare throughput
and theoretical occupancy; there is no requirement that occupancy must rise for
avoiding local memory to help. Neither a local-memory reduction nor an occupancy
increase guarantees a speedup. Do not infer A100 performance from the 4090 results
below, or infer optimality over untested assignments.

Bring back `results.csv`, `run.log`, `revision.txt` and `worktree_status.txt` (plus
`source_changes.patch` if nonempty). The log is needed to distinguish A100 variants,
MIG allocations and compiler changes. These are ordinary scalar matmul bodies,
not Tensor Core implementations, so this is a storage-policy comparison rather
than a claim to beat tuned vendor GEMM.

## Local validation

Focused registry, planner and GPU regression checks passed 133 assertions. The new
GPU checks cover Float64 precision, row/column variants, an adjoint, rectangular
subgroups, inactive output lanes, a shared intermediate, a common shared input and
a partial final block. Planner checks verify that Float64 doubles shared bytes
while preserving logical element counts and that byte budgets are enforced.

The local D32/batch5 resource run on RTX4090 compares 32 and 64 threads/block.
No Float64 throughput conclusion is drawn from this consumer GPU. Exact measured
resources are recorded in `float64_rtx4090_resources.csv`.

At both tested block sizes, compiled registers/local bytes per thread were:

| Intermediate residence | Registers | Local bytes |
|---|---:|---:|
| All registers | 255 | 480 |
| X shared | 255 | 192 |
| X/Y shared | 254 | 0 |

All six local resource candidates passed every CPU comparison with zero observed
reference error and unchanged inputs. The extra shared memory reduces theoretical
occupancy on the 4090 (8 versus 5 resident one-warp blocks, or 4 versus 2 two-warp
blocks). A100 must be measured separately: its shared-memory capacity and FP64
execution resources differ. Eliminating local memory is established locally;
a throughput benefit on A100 remains an experimental question.

Debug-mode validation passed a further nine assertions. The launcher and its
timing/CSV path were smoke-tested with a small Float32 workload; no local Float64
throughput measurements are presented as A100 evidence.
