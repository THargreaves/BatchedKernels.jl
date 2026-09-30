# Changelog

## Unreleased

### Fusion and numerical operations

- Compile large QR problems in groups of four reflectors, retaining full
  unrolling for small problems and register storage for working fragments.
- Add `BatchedRNG(seed)` and fused uniform/normal sampling for Float32/Float64
  scalars, vectors and matrices. Streams advance automatically per sampling
  launch, support reseeding/copying, and preserve draws across scheduling,
  thread-count and storage choices. Sampling-only functions accept
  `fuse(...; batch_size=N)`; ordinary calls infer the batch size from inputs.

- Automatic register-first storage is the default fusion policy. Custom
  `Assignment`s remain available; `policy=:legacy` keeps the original all-shared
  planner as a benchmark ablation baseline. Hybrid kernels can use static or
  dynamic shared-memory allocation.
- Fused kernels accept Float32 and Float64 inputs only; other element types now
  raise an `ArgumentError` under every policy.
- Add implicit stacked/block QR and residual-compression operations, preserving
  StaticArrays on CPU and returning independently planned matrix/vector/scalar
  results on GPU. Logical stacks can exceed the lane-group width.
- Support batched scalar inputs so log-likelihoods, particle weights, and backward
  normalizers can feed subsequent fused calls.
- Add scalar Joseph/SRKF forward examples with likelihood increments, normalized
  square-root backward recursions, and Gaussian candidate-weight calculations.
- Optimize QR with column-owned register fragments and shared reflector broadcasts,
  retaining scaled norms, zero-pivot handling, and joint row-sign correction.

### Integration fixes

- Label fused composite outputs from their leaf element types, e.g.
  `GaussianState{CuArray{T,1},CuArray{T,2}}` rather than
  `GaussianState{AbstractVector{T},AbstractMatrix{T}}`. Fused and hand-built
  batches with the same storage now have the same element type.
- Gather batches with `batch[idxs]`, including nested composites and host or
  device integer indices. Batched leaves receive independent storage with exact
  scalar types, shared leaves stay shared, and composite element types follow the
  gathered leaves, changing only when a view-backed leaf is copied.
- Trace triangular-root matrix products through the existing masked accessors,
  including adjoint/transpose orientations.
- Reconstruct composite types from declared field/type-parameter relationships;
  distinct model matrices no longer acquire the same traced shape merely because
  their runtime storage types match.
- Start multi-result vector lifetimes at the producing operation and preserve
  scalar outputs across fused calls. Common-only computed values receive ordinary
  intermediate storage; computation hoisting is not implemented.
- Stage scalar outputs before matrix/vector output writes so the existing scalar
  block barrier also protects the final shared-memory handoff.

### Packaging and validation

- Test sub-kernels on representative shapes instead of exhaustive size sweeps,
  reducing the full GPU suite from about 45 to 5 minutes. Run GPU tests with
  `--check-bounds=auto`: `Pkg.test()` otherwise forces bounds checks that make
  every kernel use local memory and fail the register-residency checks.
- Restrict test discovery to `test/`, excluding nested checkouts, and print live
  test names and timings. Large block-QR storage cases remain in the default
  suite and can be skipped with `BATCHEDKERNELS_TEST_EXTENDED=false`.
- Add a package API guide and an isolated integration harness against real
  GeneralisedFilters states and CPU methods. The adapter shares the scalar examples
  and preserves GeneralisedFilters' ownership of algorithms and orchestration.
- Consolidate test discovery and provide a CPU-only selection for hosted CI.
  Julia compatibility now starts at 1.11, consistent with the existing
  LinearAlgebra bound; CI targets Julia 1.11 and 1.12.
- Keep benchmark/experimental scripts separate from the application interface.
  GeneralisedFilters extension registration and complete GPU RBPF/CSMC dispatch
  are not installed by this package.
