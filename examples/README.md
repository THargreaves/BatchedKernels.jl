# Generated Kalman forward and backward steps

For actual GeneralisedFilters state types and cross-package validation, see the
[integration harness](../integration/generalised_filters/README.md). These examples
remain the shared scalar numerical definitions used by that adapter and benchmarks.

`kalman.jl` contains an ordinary scalar `joseph_kalman_step` returning
`(mean, covariance, loglikelihood_increment)`. It uses a direct Joseph covariance
update and a single innovation Cholesky for the gain, whitening and log determinant.
The gain uses `C \ HP`; tracing expands this into the existing `C.L` and `C.U`
triangular solves without forming an inverse or introducing a new subkernel.
Whitening retains `C.L \ e`, since the likelihood needs the whitened residual.

```julia
using BatchedKernels, CUDA
include("examples/kalman.jl")

# μ_data: Dx×N, P_data: Dx×Dx×N. A/Q/H/R and b/c/y are common here.
μ = BatchedCuVector(CuArray(μ_data))
P = BatchedCuMatrix(CuArray(P_data))
N = size(μ_data, 2)
args = (μ, P, SharedCuMatrix(CuArray(A), N), SharedCuVector(CuArray(b), N),
        SharedCuMatrix(CuArray(Q), N), SharedCuMatrix(CuArray(H), N),
        SharedCuVector(CuArray(c), N), SharedCuMatrix(CuArray(R), N),
        SharedCuVector(CuArray(y), N))
result = fuse(joseph_kalman_step, args...)
μ_next, P_next, increments = values(getfield(result, :components))
```

A particle-specific model matrix/vector uses `BatchedCuMatrix`/`BatchedCuVector`
instead of the corresponding shared container. Shared means common across the
current batch; its contents may change between timesteps. Replacing inputs with
matching shapes/types/lifecycles reuses the compiled specialization. Output arrays
are freshly allocated and inputs are preserved.

The same scalar function runs on ordinary Julia arrays. An application adapter can
unpack `GaussianState`/dynamics/observation fields and reconstruct its own state
from the returned mean and covariance, for example:

```julia
function particle_kalman_step(state, dynamics, observation, y)
    μ, P, ll = joseph_kalman_step(state.μ, state.Σ, dynamics.A, dynamics.b,
                                dynamics.Q, observation.H, observation.c,
                                observation.R, y)
    return GeneralisedFilters.GaussianState(μ, P), ll
end
```

The adapter is illustrative; application-specific wrappers must be resolved first. No GeneralisedFilters dependency is added
here. The application must retain its own repair and differentiation contracts;
this example performs no covariance repair and returns no reverse-mode cache.

## Supported contract

- Matching Float32 or Float64 inputs; positive state and observation dimensions
  up to 32 in the primitive geometry. Selected complete-graph validation covers
  state dimensions 3, 8, 9 and 16; resource limits still depend on shape and block size.
- Real covariance matrices, with a positive-definite innovation covariance.
  The direct covariance products do not require a Cholesky of the state covariance.
- Means, covariances, affine offsets, observations and per-particle likelihood outputs.
- Common or batched model inputs, rectangular observations, and partial/empty batches.
- Cholesky failure recovery/status reporting is not implemented. Non-positive-definite
  innovations are outside this example's supported domain; no jitter is inserted.

`symmetric_part(A)` is a traceable semantic helper for `(A+A')/2`.
`covariance_pushforward(X,A)` composes `X*A*X'` without factoring A. These are
convenient override points for application scalar functions and future specialized
bodies. The square-root variant below accepts covariance roots directly.

## Planning and measurement

The default automatic policy keeps matrix values in registers where operation
contracts allow, and uses dual-access shared storage when orientations conflict.
Vectors currently retain shared storage; scalar intermediates use registers.
Separate shared pools still serve staging and dual layouts. The implementation
reuses allocations within each pool, not the bytes of a live value across layouts.

Use `policy=:legacy` for the previous scheduler and storage path, or supply an
explicit `Assignment`. The default is 128 threads; `nthreads` is configurable.
Use `shared_memory=:dynamic` for an explicitly requested larger hybrid arena.
This policy is deterministic and cached, not autotuned or guaranteed spill-free.

Run `benchmarking/kalman/automatic_joseph.jl` for CPU-checked kernel timings,
cached allocating-call timings, cold first-call latency, and native resources.
First-call latency includes host specialization, tracing and compilation. CUDA-event
kernel samples exclude host allocation; cached call samples include synchronization.
Compare this complete graph with its own legacy baseline, not covariance-only M6.

The benchmark also includes a forced all-shared hybrid candidate using the same
operation variants, opting into dynamic shared memory when required. This isolates
storage choices from legacy compute bodies. The legacy Float64 Cholesky path is
unsupported and is explicitly omitted. Native local bytes are reported, not used
as an automatic pass/fail criterion.


## Square-root Kalman forward step

`srkf_step(μ, U, A, b, UQ, H, c, UR, y)` in the same file returns
`(mean, upper_root, loglikelihood_increment)`, with covariances represented by
`U'U`, `UQ'UQ`, and `UR'UR`. U and UR are dense upper roots; UQ can be rectangular.
Use the same batched/shared containers as above, replacing P/Q/R by their roots:

```julia
result = fuse(srkf_step, μ, U, A, b, UQ, H, c, UR, y)
μ_next, U_next, increments = values(getfield(result, :components))
```

The scalar implementation preserves static arrays and promoted scalar types on the
CPU. It follows GeneralisedFilters' square-root convention and block update: the
innovation root whitens the residual, the cross block updates the mean, and the
posterior root becomes the next state. `covariance_root_logdet(US)` supplies the
innovation covariance's log determinant without another factorization.

Two reusable helpers provide the trace boundary:

- `qr_upper_stack(A,B)` returns an upper root of `A'A+B'B`; B is square. CPU QR
  and GPU QR both put B above A, so the square block supplies the output rows.
- `qr_upper_blocks(A,B,C)` factors `[A 0; B C]` and returns `(R11,R12,R22)`.
  A is m×m, B is n×m, and C is n×n. Sign correction covers the full R row,
  including R12, with a +1 multiplier for zero diagonals.

The GPU uses private register fragments for QR working storage. Neither the
assembled matrix nor Q is materialized. Matrix result values are independently
owned, with lifetimes beginning together at the QR producer; unused blocks omit
their final stores. Output orientations can require dual shared storage under the
automatic policy. Both column-owned and row-owned QR bodies are available;
there is no register-pressure tuner or shared-scratch variant.

Fused blocks use matching Float32/Float64 and individual extents in 1:32. The
assembled dimension can exceed 32; tests include a 36×36 SRKF update and a
34×34 standalone Float64 QR. Singular state/process roots and nonempty
rectangular process roots are supported. A zero process covariance can be supplied
as a square zero root. Zero-row GPU input containers are not supported; the CPU
helper does support an empty root. Observation noise must have a nonsingular
root. Rank changes and zero pivots do not have a smooth factor derivative.

Use automatic or custom assignments for the complete SRKF graph. The legacy
scheduler/planner rejects multi-result operations with an explicit error; it
remains available for supported single-result graphs. The standalone stack QR
also has a legacy dual-storage lowering. Larger cases may need
`shared_memory=:dynamic` or fewer threads because staging buffers still use the
lane-group extent; logical block support does not override device resource limits.

`benchmarking/kalman/automatic_srkf.jl` compares automatic and all-shared hybrid
storage with the same QR bodies, checks CPU results, and records kernel/cached-call
timing, cold latency, registers, local bytes and shared bytes. Selected measurements
are recorded in [`automatic_srkf_results.csv`](../benchmarking/kalman/automatic_srkf_results.csv).
`benchmarking/kalman/sanitize_srkf.jl` exercises selected complete SRKF graphs under
CUDA sanitizers. GeneralisedFilters integration remains a separate follow-on
change; no external package source is modified here.


## Backward messages and particle weights

The backward likelihood is represented by `(B, r, logscale)`:

```math
L_t(z_t) = \exp\!\left(c_t - \tfrac12\|B_t z_t-r_t\|^2\right).
```

`B` is a dense n×n upper matrix and may be rank deficient; `r` has length n.
The normalizer includes both observation Gaussian constants and residual energy
discarded during compression. It is not a covariance root and must not be passed
to a routine that assumes positive definiteness.

The functions in `kalman.jl` follow the GeneralisedFilters CPU residual-message
design and preserve StaticArrays:

- `sqrt_backward_initialise(H,c,UR,y)` starts with the final observation.
- `sqrt_backward_predict(B,r,logscale,A,b,UQ)` integrates a future likelihood
  through `z_next = A*z + b + noise`, with covariance `UQ'UQ`.
- `sqrt_backward_update(B,r,logscale,H,c,UR,y)` multiplies by the current observation.
- `sqrt_backward_step(...)` combines prediction and update. It compresses once
  after stacking the transformed future residual with the current observation.
- `sqrt_backward_overlap(μ,U,B,r,logscale)` returns the normalized log integral
  against `N(μ,U'U)`. The four-argument form omits `logscale`.

Each function is ordinary scalar Julia code and can be fused. For example,
with batched/shared containers as in the forward examples:

```julia
message = fuse(sqrt_backward_initialise, H_T, c_T, UR_T, y_T)
B, r, logscale = values(message.components)
# A_next/b_next/UQ_next describe the transition from t to t+1;
# H_t/c_t/UR_t/y_t describe the observation at t.
message = fuse(sqrt_backward_step, B, r, logscale,
               A_next, b_next, UQ_next, H_t, c_t, UR_t, y_t)
```

`BatchedCuScalar` outputs can feed later fused calls, either directly or inside
a `BatchedStruct`. Common-only calculations within a graph currently execute per
particle into ordinary intermediate storage; the compiler does not hoist them.
A caller can precompute common observation messages if that work matters.

For either ancestor sampling or backward simulation, use
`fuse(sqrt_backward_weight, μ,U,A,b,UQ,B,r,logweight,logtransition)`.
It returns the candidate particle log weight plus the outer-state transition log
density plus the Gaussian overlap after predicting the candidate to t+1.
`kalman_backward_weight` accepts P/Q instead of U/UQ and evaluates the same
integral with covariance products and an innovation Cholesky.

Here B/r describe the fixed suffix at t+1 and μ/U (or μ/P) the candidate filter
at t. Omitting logscale is valid when that suffix is common to all candidates.
For different suffixes, include their individual logscale values before comparing
weights. Ancestor selection, trajectory storage, categorical sampling, and the
choice of precomputed versus progressively built suffix messages belong to the
application; these kernels do not implement that orchestration or draw the
conditional Gaussian state trajectory.

Two reusable helper boundaries support the backward graph:

- `qr_identity_plus(C)` returns an upper root of `I+C*C'` from implicit `[I; C']`.
- `qr_compress_residual(B,r[,C,q])` returns `(R,s,energy)` preserving squared
  residuals. GPU Householder steps transform the residual column alongside the
  matrix columns, then sum the squared residual tail. No cancellation-prone
  subtraction of norms or full assembled matrix is needed.

Each GPU matrix block has extents in 1:32; the stacked extent can be larger.
Automatic and custom assignments are supported. Backward QR has no legacy
lowering. Normalized messages allow under/overobserved states, zero process noise,
and nonempty rectangular process roots. Square zero roots represent zero noise;
zero-row GPU containers remain unsupported. Observation roots must be nonsingular
upper roots with positive diagonals. Square-root overlaps allow singular forward
roots, and covariance weights allow semidefinite P/Q. No jitter is inserted.

See `test/fusion/test_backward.jl` for independent joint-Gaussian sequence checks,
`benchmarking/kalman/automatic_backward.jl` for reproducible timings, and
`benchmarking/kalman/sanitize_backward.jl` for the CUDA sanitizer driver.
