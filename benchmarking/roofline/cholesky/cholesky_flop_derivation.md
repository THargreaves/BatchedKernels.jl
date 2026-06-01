# Flop count derivation: batched column-Cholesky

## Setup

The kernel computes the upper-triangular factor $U$ of $A = U^\top U$ for a
$D \times D$ symmetric positive-definite matrix, using one thread per column.
Thread $j$ owns column $j$; for each row $i = 1, \dots, D$, the threads with
$j \ge i$ are active. The work therefore touches exactly the upper triangle:
the index pairs $(i, j)$ with $1 \le i \le j \le D$, of which there are

$$
\binom{D+1}{2} = \frac{D(D+1)}{2}.
$$

For each active pair $(i, j)$ the kernel performs:

- an inner loop over $k = 1, \dots, i-1$, each iteration executing one
  multiply and one subtract ($A_i \mathrel{-}= A_{ki} \cdot A_k$);
- then exactly one of: a square root if $j = i$, or a division if $j > i$.

We count each multiply, add/subtract, division, and square root as one
floating-point operation. This is consistent with the convention used
throughout: a multiply-add counts as two operations. The convention is
*algorithmic* — it is independent of whether the compiler emits a fused
`FFMA` instruction.

## Multiply-add work

For a fixed pair $(i, j)$ the inner loop contributes $2(i-1)$ operations.
Summing over the triangle, grouped by column $j$ (rows $i = 1, \dots, j$):

$$
F_{\text{ma}}(D) = \sum_{j=1}^{D} \sum_{i=1}^{j} 2(i-1).
$$

The inner sum is

$$
\sum_{i=1}^{j} 2(i-1) = 2 \sum_{i=0}^{j-1} i = 2 \cdot \frac{(j-1)j}{2} = j(j-1).
$$

Hence

$$
F_{\text{ma}}(D) = \sum_{j=1}^{D} j(j-1) = \sum_{j=1}^{D} j^2 - \sum_{j=1}^{D} j.
$$

Using $\sum_{j=1}^{D} j^2 = \frac{D(D+1)(2D+1)}{6}$ and
$\sum_{j=1}^{D} j = \frac{D(D+1)}{2}$:

$$
F_{\text{ma}}(D) = \frac{D(D+1)(2D+1)}{6} - \frac{D(D+1)}{2}
= \frac{D(D+1)\big[(2D+1) - 3\big]}{6}
= \frac{D(D+1)(2D-2)}{6},
$$

which simplifies to

$$
F_{\text{ma}}(D) = \frac{D(D+1)(D-1)}{3} = \frac{D^3 - D}{3}.
$$

## Square roots

One square root is taken whenever $j = i$, i.e. once per diagonal element:

$$
F_{\text{sqrt}}(D) = D.
$$

## Divisions

One division is performed whenever $j > i$, i.e. for every
strictly-upper-triangular element:

$$
F_{\text{div}}(D) = \binom{D}{2} = \frac{D(D-1)}{2}.
$$

## Total

$$
F_{\text{chol}}(D)
= \underbrace{\frac{D^3 - D}{3}}_{\text{multiply-add}}
+ \underbrace{\frac{D(D-1)}{2}}_{\text{division}}
+ \underbrace{D}_{\text{square root}}.
$$

Expanded as a single polynomial,

$$
F_{\text{chol}}(D) = \frac{D^3}{3} + \frac{D^2}{2} + \frac{D}{6}.
$$

The three lower-order terms combine as
$-\frac{D}{3} + \frac{D(D-1)}{2} + D = \frac{D^2}{2} + \frac{D}{6}$.
The leading term $D^3/3$ is the familiar dense-Cholesky cost; the $D^2/2$
and $D/6$ terms are the exact corrections, which are not negligible at the
small $D$ studied here.

## Spot values

Useful as unit tests for the implementation:

| $D$ | $F_{\text{ma}}$ | $F_{\text{div}}$ | $F_{\text{sqrt}}$ | $F_{\text{chol}}$ |
|-----|-----------------|------------------|-------------------|-------------------|
| 2   | 2               | 1                | 2                 | 5                 |
| 4   | 20              | 6                | 4                 | 30                |
| 8   | 168             | 28               | 8                 | 204               |
| 16  | 1360            | 120              | 16                | 1496              |

## Notes for the report

The count keeps the three operation classes separate before summing. This is
deliberate: square root and division are not single hardware operations on a
GPU, even though they are single *algorithmic* operations. Counting them as
one flop each is the algorithmic convention. The consequence is that achieved
FLOP/s for Cholesky is not directly comparable, as a fraction of the FP32
FFMA peak, to that of matmul — the Cholesky flop count includes square roots
and divisions, which the FFMA peak does not model. This does not undermine
the roofline analysis, whose purpose is to identify the memory-bound versus
compute-bound regime rather than to claim a fraction of peak, but the caveat
should be stated explicitly.

This is the count for the kernel as written: column-parallel, operating on
the upper triangle, with the inner loop running exactly $i - 1$ times. It
coincides with textbook dense Cholesky because the kernel is dense Cholesky
parallelised across columns. If masking or skip logic is later added, the
count changes and this derivation must be revisited.
