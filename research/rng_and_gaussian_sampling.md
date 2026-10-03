# RNG and Gaussian sampling: current boundary and future work

## Implemented boundary

BatchedKernels owns GPU random-number generation. `BatchedRNG` uses Philox and
supports both explicit `rand`/`randn` calls inside `fuse` and ordinary
`rand!`/`randn!` fills of dense Float32/Float64 `CuArray`s. The two paths reserve
positions from the same launch counter. Randomness is addressed by logical
indices, independently of kernel scheduling. Ordinary fills can exceed the
small matrix dimensions supported inside fused calculations.

GeneralisedFilters owns the CPU/GPU routing policy. Its `CombinedRNG(cpu, gpu)`
is passed as the ordinary positional RNG argument. CPU sampling delegates to the
existing CPU generator; device array fills and fused Gaussian propagation use
the GPU child. No execution object or CPU random-number algorithm is added to
BK. GF's GPU extension unwraps the GPU child before invoking `fuse`.

Bulk resampling uniforms are generated on the GPU. Host scalar decisions, such
as systematic-resampling offsets, use the CPU child. Copying duplicates both
streams, rather than splitting them; reseeding the bundle resets both, deriving
the GPU seed from a separately seeded CPU copy. Use explicitly owned generators
such as `Xoshiro`, not `TaskLocalRNG`. Each concurrent chain needs its own bundle.

Validation covers primitive and distribution sampling with several CPU RNGs,
CPU particle Gibbs with NUTS, AbstractMCMC serial/threaded chain setup, GPU
filtering and conditional SMC, and CUDA 5/6 array fills. These are separate CPU
and GPU integration checks; they do not establish complete GPU particle-Gibbs
parameter inference or GPU differentiation support.

## Next Gaussian-sampling work

Prioritise GF's native Gaussian representations rather than making `MvNormal`
interop a prerequisite for filtering:

- Full covariance: `mu + cholesky(Sigma).L * z`.
- Upper square root with `Sigma = U'U`: `mu + U' * z`.
- Explicit, possibly rectangular covariance factor: `mu + F * z`, with as many
  standard normal components as columns of `F`.

GF already expresses these calculations in scalar sampling methods. Verify and
adapt their direct use inside `fuse`, including composition with subsequent
calculations. Reuse existing factors, prepare shared roots once where possible,
and define the zero-noise/zero-column case explicitly. Do not silently repair a
singular covariance or alter the statistical model. Sampling a conditional
Gaussian is needed for trajectory draws; RBPF forward updates should continue
to integrate the Gaussian state analytically.

An optional Distributions extension can subsequently lower supported
`rand(rng, distribution)` calls to these same traced operations. Handle two
distinct cases: adapting an existing distribution's parameters into runtime
device inputs, and constructing a distribution from traced parameters. Start
with dense multivariate normals, then explicitly add diagonal/isotropic forms.
Avoid tracing Distributions' allocation and mutation machinery merely to obtain
the same affine normal draw. Reuse BK's composite reconstruction where possible
instead of adding a parallel hierarchy of traced distribution types.

## Contracts and open design questions

- Preserve distinct random sites for distinct draws while reusing one sampled
  value when its result is reused. Keep means and factors as runtime inputs so
  changing values does not require shape-preserving recompilation.
- Maintain reproducibility for a fixed seed and call sequence. CPU/GPU bitwise
  equality, invariance to splitting batches, and invariance to changing fusion
  boundaries are not promised. Address these only if GF needs them; they require
  a more explicit logical draw-address API.
- Nonempty launches consume a reservation. Validation and compilation precede
  reservation where possible; failures after reservation do not roll it back.
  Coordinate reseeding with concurrent use. Submitted kernels retain immutable
  RNG snapshots.
- Expand beyond dense Float32/Float64 GPU-array fills only for concrete needs.
  Noncontiguous views, additional distributions, and more element types need
  deliberate dispatch and no-advance-on-rejection tests.
- GF's CPU forwarding uses two isolated Random compatibility hooks,
  `rng_native_52` and the `UInt52Raw{UInt64}` sampler. Keep Julia-version and
  dispatch-ambiguity coverage: generic forwarding alone can miss optimised bulk
  paths or recurse for MersenneTwister. Test actual MCMC clients, not only scalar
  random draws.
- Keep direct `MvNormal` integration optional. BK supplies numerical primitives;
  GF retains filtering, resampling, model validation and trajectory semantics.

## Related storage proposal

The separate shared-composite proposal is complementary: an explicit
`shared(atom, N)` helper could recursively borrow device leaves and reuse BK's
existing `BatchedStruct` reconstruction. It must not infer parameter sharing,
upload CPU arrays, repeat storage, or lift arbitrary CPU model callbacks. Mixed
shared/batched composites should continue to use explicit components. Replacing
GF's repeated selected-state arrays also requires auditing callbacks that assume
batched matrix storage; that cleanup is not automatic from adding the helper.
