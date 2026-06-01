# Flop count: triangular matmul, `flops_trmm(D)`

Computes $C = A U$ where $A$ is dense $D \times D$ and $U$ is upper
triangular $D \times D$ — the `batch_op!(*, ..., LowerTriangular(...))` calls
in the square-root Kalman kernel.

## Derivation

The kernel guards the inner contraction with `k <= d`:

```julia
for d in 1:D, i in 1:D
    tot = 0
    for k in 1:D
        if k <= d
            tot += A[i,k] * U_col[k]
```

So output column $d$ is a sum of only $d$ terms (column $d$ of $U$ has $d$
nonzeros). Per element $C[i,d]$: $d$ multiplies and $d-1$ adds — $2d-1$ flops.

Summing over rows $i = 1 \dots D$ and columns $d = 1 \dots D$:

$$
\text{flops\_trmm}(D)
= \sum_{i=1}^{D}\sum_{d=1}^{D} (2d - 1)
= D \sum_{d=1}^{D}(2d-1)
= D \cdot D^2
$$

$$
\boxed{\;\text{flops\_trmm}(D) = D^3\;}
$$

## Note

This is exactly half the leading term of a full dense matmul
($2D^3 - D^2$): the triangular structure removes the lower half of every
column's contractions. Spot checks: $D=2 \to 8$, $D=8 \to 512$,
$D=16 \to 4096$.
