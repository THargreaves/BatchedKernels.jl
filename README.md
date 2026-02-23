# BatchedKernels - Fused GPU kernels for batches of small linear algebra problems

[![Build Status](https://github.com/THargreaves/BatchedKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/THargreaves/BatchedKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Interface

```julia
vmap(f; in_type)
```

**`f`**  
A Julia function written with `LinearAlgebra` operations

**`in_type`**  
A tuple describing each argument of `f`:
- `:batched` &rarr; different value for each batch element
- `:shared` &rarr; same value for all batch elements

If `in_type` is not provided, all inputs are assumed to be `:batched`.

Returns a callable object that launches a fused CUDA kernel.

---

## Minimal example: batched matrix multiplication

```julia
using BatchedKernels
using CUDA

function matmul(A, B)
    A * B
end

matmul_vmap = BatchedKernels.vmap(
    matmul,
    in_type = (:batched, :shared), 
)

D1 = 2
D2 = 4
N = 1024

A = CUDA.rand(Float32, D1, D2, N)  # Batched matrix
B = CUDA.rand(Float32, D2, D1)  # Shared matrix

# Multiplies every matrix in batch A with matrix B
C = matmul_vmap(A, B)  # C is a (D1, D1, N) CuArray
```

---

### Supported operations (more to come)

- Matrix-matrix multiplication
- Matrix addition and subtraction
- Matrix transpose / adjoint
- Cholesky decomposition of symmetric matrices
- Triangular solves for matrices (`L \ X`, `U \ X`)
- Solves for symmetric positive definite right-hand-side (`A / Symmetric(A)`)
- Addition and subtractions with identity (e.g. `I - A`)
- Matrix-vector multiplication
- Vector addition and subtraction

---

## Benchmarking

Benchmarking scripts are provided in `benchmarking/` directory.
These can be used to compare the performance of **BatchedKernels** against
alternative implementations.

### Running the benchmark

From project root, run:
```bash
julia benchmarking/kalman/kalman_comparison.jl
```

This script will execute the Kalman filter covariance update for a range of
matrix sizes, measure the average execution time per matrix, and compare
this to other implementations. Plots will be written to `benchmarking/figs/`,
and tables will be written to `benchmarking/tables/`.