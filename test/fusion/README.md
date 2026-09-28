Ordinary `f.(args...)` and `fuse(f, args...)` use a deterministic register-first
planner. `fuse(...; policy=:legacy)` retains the original scheduler/planner.
The automatic policy propagates access requirements through wrappers and produces
values in dual-access shared storage when their orientations conflict. It uses tape
order and stable registry ordering, without timing or a register-pressure search.
Explicit assignments remain available for experiments and validated overrides.

```julia
using BatchedKernels
const BK = BatchedKernels
specs = BK.InputSpec[BK.input_spec(x) for x in args]
tape = BK.trace(f, specs)
# Inspect tape IDs, then choose supported compute variants and storage maps.
a = Assignment(tape; residences=Dict(compute_id => :single),
    orientations=Dict(compute_id => :row),
    variants=Dict(compute_id => :matmul_row), nthreads=64)
result = fuse(f, args...; assignment=a)
```

Assignments default compute values to dual storage and legacy bodies, and global
transfer staging to single storage. The input tape's node IDs remain unchanged
when transfers are normalized to `_place_input` and `_stage_output`. A single
placement aliases its staging load only with the same orientation. For column
placement, assign both the load and placement to column orientation. Unsupported
choices fail; the forced path never silently substitutes another variant.

Both shared planners preserve semantic in-place aliases and reads through old
references. The new planner uses separate single/dual pools and closed storage-owner
lifetimes, including wrappers and repeated outputs. Ordinary out-of-place results
are fresh: optional destructive input reuse is deferred. This conservative choice
can increase shared-memory use. Exact no-op staging can share an owner's allocation;
logical wrappers and orientation changes require output materialization.

Register placements and fresh register compute outputs are supported by the audited
variants. Global transfers still stage through single storage. Matching logical
orientations use direct owned-line transfers; opposite orientations use complete-group
broadcasts to materialize the result. Forced in-place operations retain their legacy
dual bodies and existing shape restrictions. Shared inputs and vectors keep their
existing residences. Explicit register-backed wrappers outside the audited accessor
domain (such as a `Symmetric` staging source) are rejected during planning. Host metadata never enters device
execution; each emitted matrix value has a concrete shape/layout binding. The
forced cache key includes the complete assignment, order, geometry and device.

`fuse(f, args...; policy=:legacy, nthreads=64)` selects legacy block geometry.
The default block size is 128 threads; explicitly supplied assignments retain their
own block size. Allocation remains part of each public call.
Shared inputs are distributed across available warps, even with more inputs than
warps. Both paths fence shared producer/consumer and slot-reuse boundaries. These
conservative boundaries are not yet a performance tuning policy.

Run focused tests with the temporary TestEnv pattern in `../memory/README.md`, using
`test/fusion` as the path. Run diagnostics off/on in separate processes. The new
cases cover owner aliasing, the separate-pool counterexample, invalid assignments,
rectangular column transfers, wrapped/repeated outputs, partial batches, four shared
inputs at 64 threads, and public return inference. GPU resource checks compare the
planner's aligned byte budget with compiled static shared memory, allowing unused
tail padding and dead allocation elimination. Use memcheck, racecheck and synccheck for the focused GPU item.

`HybridPlannerOutput.peak_register_elements` counts live per-lane matrix elements,
including overlapping inputs and fresh outputs. Row orientation contributes the row
extent and column orientation the column extent. Wrappers share their parent's count.
This excludes compiler temporaries, pointers and allocation overhead, and is not a
prediction of physical registers. `CUDA.registers(kernel)` and `CUDA.memory(kernel)`
inspect the assembled kernel; a register assignment can still spill. The M6 benchmark
admits only CPU-correct configurations with zero compiled local memory.

The selected register tests cover rectangular source-lane participation, mixed
register/shared computation, wrapped/repeated outputs, partial batches, a Cholesky/two
solve pipeline and public output inference. Run both accessor modes; resource gates
apply to production builds. See `benchmarking/kalman/hybrid_m6.jl` for the full-fusion
performance comparison.

Explicit hybrid assignments can opt into a larger per-block shared arena:

```julia
result = fuse(f, args...; assignment=a, shared_memory=:dynamic)
```

Static remains the default. Dynamic mode uses constant aligned offsets, checks
planner bytes against the device's opt-in capacity and configures launch bytes.
The mode is part of the cache key. Debug accessors retain device arena bounds
checks; production relies on the validated host launch contract. The focused
`test_dynamic_shared.jl` covers inference, cache separation, mixed region types,
partial blocks, unchanged inputs and oversized-arena rejection.

The complete Joseph forward example is `examples/kalman.jl`; it returns mean,
covariance and one log-likelihood increment per particle. `test_automatic.jl`
checks independent CPU references, rectangular observations, common/batched model
inputs, Float32/Float64, empty/partial/full batches, input preservation and inferred
outputs. Matrix-vector products and triangular-vector solves consume audited matrix
layouts; vectors retain shared storage. Scalar reductions use complete group masks,
including D=32 and logical observations smaller than the group.

Run only this item file through a directory filter (passing a file path to
`run_tests` does not discover test items):

```julia
TestItemRunner.run_tests("test/fusion";
    filter=ti -> occursin("test_automatic.jl", ti.filename), verbose=true)
```


Structured R-only QR uses tuple-valued `CallNode`s with matrix/vector/scalar `ResultNode`
projections. The producer executes once; projections emit no code and own fresh
storage. Every result interval begins at the producer's schedule position, even
when a custom order delays the projection. Variant output-access tuples specify
each block independently. Ordinary single-result operation contracts are unchanged.
`Assignment` selects the registered QR variant for these calls by default (dual
outputs); automatic assignment chooses register/dual placements by access demand.
The legacy scheduler/planner rejects multi-result tapes explicitly.

`test_block_qr.jl` covers static CPU inference, promoted element types, result
lifetimes, independent output storage policies, unequal/padded/full-warp blocks,
assembled dimensions over 32, scaled norms, zero pivots, rank-deficient roots,
rectangular process noise, complete SRKF likelihoods and repeated zero-noise steps.


QR performance variants additionally exercise column-owned `:row` and row-owned
`:col` contracts, including custom shared assignments and the original legacy
stack lowering. Numerical edge coverage includes genuinely subnormal columns
(Float32 1e-40, Float64 1e-310), huge/tiny normal scales, exact Float64 magnitude
ordering, masked triangular inputs and rescaled Gram comparisons. Existing SRKF
recursion and mean/root/likelihood tolerances are retained.


`test_backward.jl` adds residual compression and identity-plus QR, StaticArrays
inference, normalized multi-step likelihoods checked against an independently
assembled joint Gaussian, zero/rank-deficient process noise, and shared suffix
weights for both forward covariance representations. It covers scalar inputs,
composite message recurrence, common-only calculations, delayed vector projections,
unused results, rectangular blocks, and stacks larger than a lane group.
`backward_reference.jl` supplies the dense reference only; no application package
is required to run these tests.

## GeneralisedFilters integration regressions

`test_composite_shapes.jl` verifies independent type parameters for equally typed
model matrices with different logical shapes. `test_wrapped_products.jl` checks
triangular-root products, including adjoint/transpose orientation and nonzero
data in the masked triangle. The separate
[real-package harness](../../integration/generalised_filters/README.md) validates
wrapped state recursion and likelihood conventions against GeneralisedFilters.

The package test entry point discovers all test items. CPU-only environments run
`:cpu` tests; `BATCHEDKERNELS_TEST_CPU_ONLY=true` selects that path explicitly.
