# GeneralisedFilters integration boundary

This directory is a merge-preparation harness, not an installed package extension.
It loads the actual GeneralisedFilters package and wraps the scalar definitions
in [`examples/kalman.jl`](../../examples/kalman.jl) with its real state and likelihood
types. No GeneralisedFilters methods or files are replaced. The adapter has no
compiler-internal dependencies and does not add GeneralisedFilters to the
BatchedKernels dependency graph.

## Reproduce

The current GeneralisedFilters checkout requires Julia 1.12.7 and CUDA.jl 5.
Resolve a separate environment so the main BatchedKernels development environment
can continue to exercise CUDA.jl 6:

```sh
julia +1.12.7 --startup-file=no --project=integration/generalised_filters \
  integration/generalised_filters/setup.jl /path/to/GeneralisedFilters /path/to/SSMProblems
julia +1.12.7 --startup-file=no --project=integration/generalised_filters \
  integration/generalised_filters/runtests.jl
```

The SSMProblems path is optional if a compatible version is available from the
registry. Paths are recorded only in the ignored local Manifest.toml. Setup
modifies this integration environment, not the target package. Set
`BATCHEDKERNELS_TEST_CPU_ONLY=true` for the CPU checks alone. The test log reports
the Julia/CUDA.jl versions and whether GPU checks actually ran.

## Contracts to preserve during the merge

| GeneralisedFilters value | Fused representation / rule |
| --- | --- |
| `GaussianState(μ, Σ)` | Composite of batched mean and full covariance arrays |
| `SqrtGaussianState(μ, U)` | Composite with a real `UpperTriangular` wrapper; covariance is `U'U` |
| `CovarianceFactor(F)` | Covariance is `F*F'`; provide process root `UQ=F'`, including nonempty rectangular or zero factors |
| Plain noise covariance | Prepare its upper Cholesky root before the square-root boundary, once if common |
| Observation root `UR` | Square, nonsingular, upper, positive diagonal; covariance `UR'UR` |
| `SqrtInformationLikelihood(B,r,logscale)` | Complete residual likelihood `exp(logscale - ||B*z-r||²/2)` |
| Forward result | `(state, loglikelihood_increment)`; the increment updates particle weights exactly once |
| Candidate weight | Filtering log weight + outer transition log density + Gaussian suffix overlap |

`CovarianceFactor` observation factors need conversion to a canonical upper root
before this interface; transposing an arbitrary square factor does not make it
upper triangular. Process factors need no such restriction. A zero-column factor
must be represented by a square zero root on GPU; zero-row GPU matrices are not
supported. Inputs to a fused graph use one matching floating-point type. CPU
GeneralisedFilters may promote types more broadly (including its current Float32
SRKF log-likelihood calculation).

Keep states, model arrays, log weights and messages in structure-of-arrays device
storage across calls. Wrap owned arrays with BatchedKernels containers; do not
materialize an array of host state objects every step. The `pack` and `host`
functions in the tests are fixture utilities, not the intended production data
path. Generated composite outputs can feed the next call directly:

```julia
# After including adapter.jl; the inputs here are already batched/shared containers.
const GK = GeneralisedFiltersKernels
result = fuse(GK.root_step, state, A, b, UQ, H, c, UR, y)
state, increments = values(result.components)
result = fuse(GK.root_step, state, A_next, b_next, UQ_next, H_next, c_next, UR_next, y_next)

message = fuse(GK.backward_start, H, c, UR, y)
message = fuse(GK.backward_step, message, A, b, UQ, H, c, UR, y)
weights = fuse(GK.root_weight, candidate_states, A, b, UQ, suffix_message,
               filtering_logweights, outer_transition_logdensities)
```

The suffix in a candidate-weight call is at t+1 and the candidate forward state
is at t. Its omitted `logscale` must be common to all candidates; restore it when
comparing different suffixes. `GK.normalized_overlap` retains it. QR row signs
can differ from GeneralisedFilters' existing implementation: compare likelihoods
or `(B'B, B'r, logscale - r'r/2)`, not individual factors. There is no process
covariance inverse and no backward jitter.

## Merge sequence

1. Keep GeneralisedFilters' existing CPU methods and StaticArrays behavior as the
   reference. Introduce small semantic helper boundaries for implicit QR stacks,
   residual compression, symmetrization and covariance products. The corresponding
   BatchedKernels helpers are listed in the [package README](../../README.md).
2. Add an optional GeneralisedFilters extension activated by BatchedKernels and
   CUDA. That extension owns the adapter and batch dispatch. The adapter here
   demonstrates the state/result shapes and calls only public BatchedKernels APIs;
   it does not install such an extension automatically. Avoid a mandatory CUDA
   dependency in the GeneralisedFilters CPU path.
3. Resolve time/outer-state-dependent model components before the fused boundary.
   Share common roots and arrays; batch only particle-dependent quantities. Use
   named functions, and preserve the automatic/legacy/custom assignment choices.
4. Connect the generated likelihood increments and candidate weights to the
   existing RBPF/CSMC orchestration. Preserve APF corrections already present in
   filtering weights; do not add a lookahead term again. Keep AS suffix
   precomputation and BS suffix updates in their respective algorithms.
5. Validate complete filter and trajectory-sampling runs in GeneralisedFilters
   after that wiring. The checks here establish Gaussian numerical and container
   compatibility, not a complete GPU particle-filter implementation.

The covariance adapter uses the no-repair Joseph update. Explicit repair policies
and the analytic `kalman_step_cached`/Mooncake pullback remain owned by
GeneralisedFilters; the fused forward adapter does not return the reverse-pass
cache. No GPU differentiation support is claimed. Automatic model resolution,
categorical sampling, particle gathering, Gaussian trajectory draws, and history
storage are outside this harness.

## Validation coverage

The harness compares against `kalman_step`, `srkf_predict`/`srkf_update`,
`backward_initialise`/`backward_predict`/`backward_update`, and
`compute_marginal_predictive_likelihood` from the loaded GeneralisedFilters.
It exercises Float32/Float64, CPU StaticArrays, partial GPU batches, singular and
rectangular process factors, repeated wrapped-state calls, normalized messages,
relative candidate weights with a shared suffix, and dynamic shared-memory output. Existing package
tests retain the independent joint-Gaussian and larger-shape coverage.

Validated on Julia 1.12.7, CUDA.jl 5.11.3, RTX 4090: **1,080 assertions passed**,
with GPU checks enabled. The main package environment passes 1,998 fusion and 289 public fused-kernel
assertions on CUDA.jl 6.1.0 (Julia 1.12.6).

The consolidation also checks the package CPU-only `Pkg.test()` path (496
assertions). The exhaustive standalone legacy QR shape sweeps were not completed
in this pass; they remain part of the full GPU package test command.
