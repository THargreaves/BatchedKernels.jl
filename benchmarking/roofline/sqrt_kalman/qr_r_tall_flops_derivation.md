# Flop count: tall Householder QR (R-only), `flops_qr_r_tall(D1, D2)`

Householder QR of a $D_1 \times D_2$ matrix with $D_1 \ge D_2$ (R-only output).
Generalises `flops_qr_r` (the square $D \times D$ case) to a rectangular
input. In the square-root Kalman kernel the predict-step QR is $2D \times D$,
i.e. $D_1 = 2D$, $D_2 = D$.

## Setup

The outer loop runs over columns $j = 1, \dots, D_2$ — **$D_2$ steps**. (The
square kernel stops at $D-1$; a tall matrix still has rows to eliminate below
the diagonal in the last column, so the loop runs the full $D_2$.)

At step $j$:

- **active rows** $m = D_1 - j + 1$ — rows $i \ge j$ that the reflector
  touches. Runs $D_1 \to D_1 - D_2 + 1$.
- **trailing columns** $n = D_2 - j + 1$ — columns $t \ge j$ updated by the
  reflector. Runs $D_2 \to 1$.

For the square case $D_1 = D_2$ gives $m = n$, recovering the original.

In the $2D \times D$ kernel each thread owns two rows (top block row $i$,
bottom block row $i+D$); the top block contributes the guarded rows $i \ge j$
and the bottom block contributes all $D$ rows, so the active count is
$m = (D - j + 1) + D = 2D - j + 1$, consistent with $D_1 - j + 1$ at
$D_1 = 2D$.

## Counting conventions

Identical to `flops_qr_r`: each multiply, add, subtract, divide and `sqrt` is
$1$ flop; a `warp_reduce_sum` over $m$ values is $m-1$ adds; broadcast
(`shfl`) is $0$. Four tallies: $\mathrm{AS}$ (add/sub), $\mathrm{ML}$
(multiply), $\mathrm{DV}$ (divide), $\mathrm{SQ}$ (sqrt).

## Per-step count (active rows $m$, trailing columns $n$)

1. **Column norm** — $m$ squares, reduction $m-1$ adds.
   $\mathrm{ML}\!+\!=\!m$, $\mathrm{AS}\!+\!=\!m-1$.
2. **Reflector entry** — $1$ sqrt, $1$ multiply, $1$ subtract.
   $\mathrm{SQ}\!+\!=\!1$, $\mathrm{ML}\!+\!=\!1$, $\mathrm{AS}\!+\!=\!1$.
3. **Broadcast `v1`** — $0$.
4. **Normalise** — $m$ divides. $\mathrm{DV}\!+\!=\!m$.
5. **Tau** — $m$ squares, $m-1$ adds, $1$ divide.
   $\mathrm{ML}\!+\!=\!m$, $\mathrm{AS}\!+\!=\!m-1$, $\mathrm{DV}\!+\!=\!1$.
6. **Broadcast `tau`** — $0$.
7. **`tau_v`** — $m$ multiplies. $\mathrm{ML}\!+\!=\!m$.
8. **Trailing update** — $n$ columns; per column $m$ mul $+ (m-1)$ add for
   $w_t$, then $m$ mul $+ m$ sub for the rank-1 update.
   Per column $\mathrm{ML}\!+\!=\!2m$, $\mathrm{AS}\!+\!=\!2m-1$;
   over $n$ columns $\mathrm{ML}\!+\!=\!2mn$, $\mathrm{AS}\!+\!=\!n(2m-1)$.

### Per-step subtotals

$$
\mathrm{AS}(m,n) = 2(m-1) + 1 + n(2m-1), \qquad
\mathrm{ML}(m,n) = 3m + 1 + 2mn
$$
$$
\mathrm{DV}(m) = m + 1, \qquad \mathrm{SQ} = 1
$$

## General closed form

Summing each per-step subtotal over $j = 1, \dots, D_2$ with
$m = D_1 - j + 1$ and $n = D_2 - j + 1$ gives (verified symbolically):

$$
\text{AS}(D_1,D_2) = D_1 D_2^2 + 3 D_1 D_2
\tfrac{1}{3}D_2^3 - \tfrac{3}{2}D_2^2 - \tfrac{1}{6}D_2
$$

$$
\text{ML}(D_1,D_2) = D_1 D_2^2 + 4 D_1 D_2
$$

$$
\tfrac{1}{3}D_2^3 - \tfrac{3}{2}D_2^2 + \tfrac{17}{6}D_2
$$

$$
\text{DV}(D_1,D_2) = D_1 D_2 - \tfrac{1}{2}D_2^2 + \tfrac{3}{2}D_2,
\qquad
\text{SQ}(D_2) = D_2
$$

Their sum is the total algorithmic flop count:

$$
\boxed{\;\text{flops_qr_r_tall}(D_1, D_2) = 
2 D_1 D_2^2 + 8 D_1 D_2 - \tfrac{2}{3} D_2^3- \tfrac{7}{2} D_2^2 + \tfrac{31}{6} D_2\;}
$$

valid for any $D_1 \ge D_2$.

## Substitution 1 — the $2D \times D$ predict-step QR

Setting $D_1 = 2D$, $D_2 = D$:

$$
\text{AS} = \tfrac{5}{3}D^3 + \tfrac{9}{2}D^2 - \tfrac{1}{6}D, \qquad
\text{ML} = \tfrac{5}{3}D^3 + \tfrac{13}{2}D^2 + \tfrac{17}{6}D
$$
$$
\text{DV} = \tfrac{3}{2}D^2 + \tfrac{3}{2}D, \qquad \text{SQ} = D
$$

$$
\boxed{\;\text{flops_qr_r_tall}(2D, D) =
\tfrac{10}{3}D^3 + \tfrac{25}{2}D^2 + \tfrac{31}{6}D\;}
$$

## Substitution 2 — verification against the square case

Setting $D_1 = D_2 = D$:

$$
\text{flops_qr_r_tall}(D, D) =
\tfrac{4}{3}D^3 + \tfrac{9}{2}D^2 + \tfrac{31}{6}D
$$

The earlier square derivation `flops_qr_r` loops $j = 1, \dots, D-1$ and gives

$$
\text{flops_qr_r}(D) = \tfrac{2}{3}D(D+1)(2D+1)
+ \tfrac{5}{2}D(D+1) + 2D - 11.
$$

The general form here loops $j = 1, \dots, D_2 = D$ — one extra step. That
final $j = D$ step has $m = n = 1$: $\text{AS}=2,\ \text{ML}=6,\
\text{DV}=2,\ \text{SQ}=1$, total $11$. Therefore

$$
\text{flops_qr_r_tall}(D, D)
= \text{flops_qr_r}(D) + 11
= \tfrac{4}{3}D^3 + \tfrac{9}{2}D^2 + \tfrac{31}{6}D.
$$

The two agree exactly up to the loop-bound difference — the per-step
arithmetic is identical, confirming the general derivation.

## Validation

**Leading term.** $2 D_1 D_2^2 - \tfrac{2}{3}D_2^3$; at $D_1 = D_2 = D$ this is
$\tfrac{4}{3}D^3$, the textbook Householder QR leading term. For the tall
$2D \times D$ case the leading term is $\tfrac{10}{3}D^3$.

**Spot checks** (brute per-step sum vs closed form, $2D \times D$ case):

| $D$ | AS | ML | DV | SQ | total |
|-----|----|----|----|----|-------|
| $2$  | $31$   | $45$   | $9$   | $2$  | $87$    |
| $4$  | $178$  | $222$  | $30$  | $4$  | $434$   |
| $8$  | $1140$ | $1292$ | $108$ | $8$  | $2548$  |
| $16$ | $7976$ | $8536$ | $408$ | $16$ | $16936$ |

All match the closed forms.

## Use in the square-root Kalman flop count

The SR-KF kernel performs:

- predict-step QR: $2D \times D$ → `flops_qr_r_tall(2D, D)` =
  $\tfrac{10}{3}D^3 + \tfrac{25}{2}D^2 + \tfrac{31}{6}D$.
- update-step QR: $2D \times 2D$ → `flops_qr_r_tall(2D, 2D)`
  (square, the known case at dimension $2D$).
- plus the two `batch_op!(*, ...)` matmuls and the layout transfers, each a
  standard $D \times D$ operation with a known count.

