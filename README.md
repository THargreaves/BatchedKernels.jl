# BatchedKernels.jl

BatchedKernels is a Julia package designed to automatically produce fused CUDA kernels
for performing sequences of linear algebra operations on large batches of small ($D<32$)
matrices/vectors. It's behaviour is somewhat similar to JAX's `vmap` combined with `jit`, 
but typically results in much faster kernels, especially when general linear algebra 
operations beyond multipliaction or element-wise operations are included.

The package is under development. It requires Julia 1.11 or newer and a
CUDA-capable GPU for fused execution. From a checkout, install dependencies with:

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## A Simple Example

BatchedKernels takes a generic Julia function that performs a sequence of linear algebra
operations on a singleton input of matrices, vectors, or scalars, and automatically converts
it into a single batched CUDA kernel, fusing the operations to reduce memory traffic and kernel 
launch overhead.

```julia
using BatchedKernels, CUDA

update(A, x, b) = A * x + b

N = 1000
A = SharedCuMatrix(CUDA.rand(Float32, 3, 3), N)
xs = BatchedCuVector(CUDA.rand(Float32, 3, N))
b = SharedCuVector(CUDA.rand(Float32, 3), N)

ys = fuse(update, A, xs, b)
size(ys.data)  # (3, 1000)
# update.(A, xs, b) uses the same default fusion path.
```

Here each entry has its own `x`, while all entries use the same `A` and `b`.
The wrappers reference existing arrays. `fuse` allocates the output and launches
the kernel.

| Container | Backing array | Meaning |
| --- | --- | --- |
| `BatchedCuMatrix(data)` | `m × n × N` | One matrix per entry |
| `BatchedCuVector(data)` | `d × N` | One vector per entry |
| `BatchedCuScalar(data)` | Length `N` | One scalar per entry |
| `SharedCuMatrix(data, N)` | `m × n` | One matrix used by every entry |
| `SharedCuVector(data, N)` | Length `d` | One vector used by every entry |

Input containers must agree on `N`. `BatchedStruct(T, components)` groups named
component batches into a batch of structs; `SharedValue(value, N)` supplies a
shared literal field. A function returning a tuple produces a `BatchedStruct`;
use `values(result.components)` to unpack its batched outputs. Outputs can feed
subsequent `fuse` calls directly.

`batch[idxs]` gathers entries into a new batch, applying the same integer indices
(on the host or device) to every field of a `BatchedStruct`. Gathered batched
fields get their own storage; shared fields stay shared.

## Random sampling

Pass a `BatchedRNG` explicitly. Sampling then runs inside the fused kernel:

```julia
using BatchedKernels, CUDA, Random

perturb(rng, x) = x + randn(rng, eltype(x), size(x, 1))

rng = BatchedRNG(123)
xs = BatchedCuVector(CUDA.zeros(Float32, 3, 1000))
ys = fuse(perturb, rng, xs)
zs = fuse(perturb, rng, xs)  # fresh samples
```

`rand(rng, T, dims...)` and `randn(rng, T, dims...)` support Float32/Float64
scalars, vectors and matrices. Dimensions must be known during tracing and lie
in `1:32`. Omitting `T` selects Float64. The same function can use an ordinary
Julia RNG when called directly on CPU arrays.

The RNG advances automatically once per nonempty sampling launch. Tracing,
compilation, validation failures, empty batches and unused RNG arguments do not
advance it. Use `copy(rng)` to preserve its current position in an independent
RNG, or `Random.seed!(rng, 123)` to restart. Batch size comes from the input
containers; for sampling-only functions, supply `fuse(sample, rng; batch_size=N)`.

The same seed and sequence of calls reproduce samples within a package version.
Changing thread count, scheduling or storage layout preserves samples; splitting
one batch across several calls changes them. CPU RNG sequences and bitwise normal
samples across devices or versions are not guaranteed. Concurrent callers reserve
distinct stream positions, but their order is not fixed.

Always pass the RNG to sampling calls. Implicit `rand()` or `randn()` executes
during host tracing and can become a cached constant. Distribution objects,
`rand!`/`randn!`, and random-dependent control flow are unsupported.

## Supported code and limits

- Fused numerical inputs and intermediates use one matching type: Float32 or
  Float64 (mixed precision is not supported). 
- Supported operations include matrix products and addition/subtraction, matrix-vector
  products, vector addition/subtraction, triangular solves, Cholesky, QR,
  `dot`, `sum(abs2, x)`, and scalar arithmetic. Support is specific to operand
  types: for example, triangular matrix solves accept upper or lower factors,
  while triangular vector solves currently accept lower factors.
- `logabsdet` accepts real upper/lower triangular matrices, including
  adjoint/transpose parents and Cholesky `.L`/`.U` factors, and returns the
  standard `(log_magnitude, sign)` tuple. Negative, zero, and nonfinite diagonals
  follow Julia's triangular semantics. Arbitrary dense `logabsdet` is not supported.
- Use named functions with supported array operations. Capturing closures,
  callable structs, arbitrary scalar indexing, elementwise array broadcasts,
  and branches on computed values are unsupported.
- General mutation is unsupported. Explicit `cholesky!` and matrix `ldiv!`
  have dedicated implementations with shared storage and narrower shape limits.
  Ordinary out-of-place calculations preserve input arrays.

The default storage policy prefers registers and uses shared memory where needed.
The first call for a new function or input shape includes compilation; later
calls reuse the kernel. Changing only batch size or RNG seed does not require a
new kernel. See the [developer notes](docs/development.md) for the compiler path,
launch options and tests.

## Further reading

- [Kalman examples](examples/README.md): forward/backward steps, covariance and
  QR helper contracts, and particle-weight calculations.
- [Kalman benchmarks](benchmarking/kalman/README.md) and
  [matrix multiplication benchmarks](benchmarking/matmul/README.md).
