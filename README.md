# BatchedKernels.jl

Generate a CUDA kernel from a scalar Julia function over small matrices, vectors,
and scalars. The same scalar code can run on CPU arrays or StaticArrays. Fused
kernels support Float32 and Float64 only (other element types are rejected);
individual matrix extents
are currently bounded by a 32-lane group. QR can operate on implicit stacks larger
than that group.

```julia
using BatchedKernels, CUDA

product(A, B) = A * B
A = BatchedCuMatrix(CUDA.rand(Float32, 3, 3, 1000))
B = SharedCuMatrix(CUDA.rand(Float32, 3, 3), 1000)
C = fuse(product, A, B)  # C.data is a 3×3×1000 CuArray
# product.(A, B) uses the same default fusion path.
```

Use named singleton functions at the fusion boundary. Closures, callable structs,
arbitrary scalar indexing, and mutation inside the traced function are not
supported, apart from explicit in-place `cholesky!` and `ldiv!`, which keep their
shared-memory implementation. This is a compiler for a defined set of array operations, rather than
a general Julia-to-GPU transformation.

## Public integration surface

| API | Role |
| --- | --- |
| `BatchedCuMatrix`, `BatchedCuVector`, `BatchedCuScalar` | Wrap device arrays with particle index on the last axis |
| `SharedCuMatrix`, `SharedCuVector` | Reuse one device array across a batch |
| `BatchedStruct`, `SharedValue` | Composite states and literal fields |
| `fuse(f, args...; ...)` | Allocate results and launch a generated kernel |
| `Assignment`, `automatic_assignment` | Explicit storage/orientation choices for an advanced caller |
| `symmetric_part`, `covariance_pushforward` | Scalar matrix operations with supported tracing |
| `qr_upper_stack`, `qr_upper_blocks` | Implicit stacked/block R-only QR |
| `qr_identity_plus`, `qr_compress_residual` | Gaussian residual propagation and compression |
| `covariance_root_logdet` | Log determinant of a covariance represented by an upper root |

`fuse` defaults to `policy=:auto`, which prefers register intermediates and uses
shared memory for staging and incompatible access orientations. `policy=:legacy`
retains the earlier all-shared scheduler and planner as an ablation baseline for
benchmarks; it is not intended for applications. An explicit
`assignment` selects the custom hybrid path. QR with multiple results requires
automatic or custom assignment. `nthreads` and `shared_memory=:static/:dynamic`
control launch geometry and shared allocation; device resource limits still apply.
The planner, tape, accessors, cache, and `_ensure_compiled!` are implementation
details; application integrations should use `fuse`.

## Kalman and GeneralisedFilters

[Scalar examples and contracts](examples/README.md) cover Joseph and square-root
forward steps (including log-likelihood increments), normalized backward
likelihoods, and ancestor/backward-simulation candidate weights. These are
application reference functions, not an exported filtering framework.

[GeneralisedFilters integration](integration/generalised_filters/README.md)
provides a thin reference adapter and an isolated test environment that loads the
real package. It checks CPU StaticArrays and GPU results against its existing
algorithms. GeneralisedFilters continues to own model resolution, particle
weights, resampling, trajectory storage, covariance repair, and differentiation.
BatchedKernels has no dependency on GeneralisedFilters.

The [readiness report](benchmarking/kalman/SMALL_KALMAN_READINESS.md) records the
implementation history. Current measurements are in the
[SRKF](benchmarking/kalman/SRKF_PERFORMANCE.md) and
[backward](benchmarking/kalman/BACKWARD_PERFORMANCE.md) reports. Benchmark scripts
and experimental variants are development tools, not required by an application.

## Testing

Julia 1.11 or newer is required by the package dependency bounds. The separate
GeneralisedFilters harness follows that package's stricter version requirements.

```sh
julia --project -e 'using Pkg; Pkg.test()'
BATCHEDKERNELS_TEST_CPU_ONLY=true julia --project -e 'using Pkg; Pkg.test()'
```

With a functional CUDA GPU, the first command discovers all test items, including
sub-kernel, layout, compiler, and generated-kernel tests. The standalone legacy QR
tests include exhaustive shape sweeps and can take substantially longer than the
fusion suites. Without a GPU, the command runs the
explicitly tagged CPU tests. Hosted CI runs on Julia 1.11 and 1.12; a passing CPU
job does not establish GPU correctness. Use the GPU suite and the sanitizer
drivers under `benchmarking/kalman` for changes to generated device code.
