# Developer notes

See the [README](../README.md) for usage and current limits.

## From function to kernel

1. **Describe inputs and trace.** Container types supply element types, shapes
   and whether a value varies across the batch. The tracer calls the user's
   function with placeholder matrices, vectors, scalars and RNGs. Supported
   operations append nodes to a tape; they do not calculate array values.
   Start with [trace.jl](../src/fuse/trace.jl) and
   [overloads.jl](../src/fuse/overloads.jl).
2. **Choose operations and storage.** The automatic planner selects operation
   variants and places matrix intermediates in registers or shared memory.
   Access orientation and value lifetimes determine transfers and storage reuse.
   Vectors use shared storage; scalars use registers. See
   [automatic.jl](../src/fuse/automatic.jl) and
   [assignment.jl](../src/fuse/assignment.jl).
3. **Generate code.** The generator emits a Julia kernel containing loads,
   operation bodies, synchronization and stores. CUDA.jl compiles it for the
   device. See [codegen.jl](../src/fuse/codegen.jl) and the operation registry in
   [variants.jl](../src/fuse/variants.jl).
4. **Allocate and launch.** The public call allocates output arrays, launches the
   kernel and wraps its results. [broadcast.jl](../src/fuse/broadcast.jl) contains
   `fuse`, broadcast dispatch, caching and launch handling.

Tracing executes ordinary Julia code on the host. Only supported operations on
traced values become kernel operations; unrelated host computation and side
effects can happen during tracing. Avoid side effects in fused functions.

Random sampling uses Philox4x32-10 with addresses formed from the launch counter,
batch index, sampling call and logical element. The host reserves stream positions
after compilation and resource checks, just before launch. Reservations are
synchronized and are not rolled back after launch/device errors. See
[random.jl](../src/random.jl) for state and sampling, and
[fuse/random.jl](../src/fuse/random.jl) for tracing and emission.

## Compilation and launch options

Compiled kernels are cached within the Julia process by function, input
specialization, storage choices, launch settings and device. Shapes and numerical
types affect specialization; batch size, array contents and RNG state do not.
`SharedValue` contents are trace-time literals and do affect specialization.

Use `fuse` for application code. The tape, planner and kernel cache are internal.
`nthreads` defaults to 128 (or the explicit assignment's thread count) and must
be a multiple of 32 in `32:1024`.
`shared_memory=:dynamic` permits an arena up to the device's opt-in shared-memory
limit; `:static` is the default. The automatic policy does not search for the
fastest configuration or guarantee that register values will avoid spilling.

An explicit `Assignment` overrides storage and variant choices. `policy=:legacy`
retains the earlier all-shared path for benchmark comparisons; it supports fewer
operations and cannot use dynamic shared memory. Multi-result QR needs automatic
or explicit assignment. See the [compiler and storage notes](../test/fusion/README.md)
for those experiments.

Measure first-call compilation separately from repeated calls. CUDA launches are
asynchronous: synchronize when measuring completed work. Public `fuse` timings
include output allocation; kernel-only timings measure a different cost. Existing
[benchmark scripts](../benchmarking/kalman/README.md) report these separately.

## Tests

Run from the repository root:

```sh
julia --project -e 'using Pkg; Pkg.test()'
BATCHEDKERNELS_TEST_CPU_ONLY=true julia --project -e 'using Pkg; Pkg.test()'
```

Each test item prints its name and elapsed time. With CUDA available, the first
command runs all items, including extensive shape/layout sweeps. Without CUDA,
or with the environment variable above, only `:cpu` items run. Hosted CI uses
Julia 1.11 and 1.12 and runs the CPU selection.

To skip the largest block-QR storage cases during local development:

```sh
BATCHEDKERNELS_TEST_EXTENDED=false julia --project -e 'using Pkg; Pkg.test()'
```

Other shape sweeps still run. The default full selection retains all cases.

For generated-code changes, run the relevant GPU tests as well as CPU checks.
The [fusion tests](../test/fusion/README.md) describe focused selections; the
[memory tests](../test/memory/README.md) describe accessor diagnostics.
Sanitizer drivers live in [benchmarking/kalman](../benchmarking/kalman/README.md).
The [GeneralisedFilters harness](../integration/generalised_filters/README.md)
has its own environment and version requirements.
