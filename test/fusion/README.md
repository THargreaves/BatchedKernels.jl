The forced shared-storage path is an explicit integration surface. Ordinary
`f.(args...)` continues to use the corrected legacy planner.

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

Register assignments remain M6 work. Forced in-place operations retain their legacy
dual bodies and existing shape restrictions. Host metadata never enters device
execution; each emitted matrix value has a concrete shape/layout binding. The
forced cache key includes the complete assignment, order, geometry and device.

`fuse(f, args...; nthreads=64)` selects the legacy path with explicit block geometry.
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
