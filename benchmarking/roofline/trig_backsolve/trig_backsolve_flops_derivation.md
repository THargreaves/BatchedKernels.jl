# Flop count: triangular back-solve, `flops_trig_backsolve(D)`

## Setup

Solves $C = U \backslash A$ where $U$ is upper-triangular and $A$, $C$ are dense $D \times D$. One thread per right-hand-side column: thread owns column $d$ of the RHS and computes column $d$ of the result by back-substitution.

Per column, the kernel runs $i$ from $D$ down to $1$:

$$
x_i = \frac{A_{i,d} - \sum_{j>i} U_{i,j}\,x_j}{U_{i,i}}
$$

The inner loop is written `for j in 1:D` with an `if j > i` guard, so although it is unrolled over the full range, only the $j > i$ cases do work. For row $i$ there are $D - i$ such terms.

## Counting conventions

- **Algorithmic flops.** Each multiply, add, subtract, divide, `sqrt` counts as $1$ flop, summed with equal weight in the final formula.
- **Counted separately by type.** Add/sub, multiply, divide, and `sqrt` are tallied separately below, because their hardware costs differ sharply (add/sub/mul issue at full FP32 throughput; divide is a multi-instruction sequence). The split shows how much of the work is cheap multiply-add versus expensive divide.
- Tallies: $\mathrm{AS}$ = adds + subtracts, $\mathrm{ML}$ = multiplies, $\mathrm{DV}$ = divides, $\mathrm{SQ}$ = square roots.

## Per-column count

Back-substitution for one RHS column. For each row $i$ (from $D$ down to $1$):

**Inner sum** — $\sum_{j>i} U_{i,j}\,x_j$, which has $D - i$ terms:
- each term $U_{i,j}\,x_j$ is $1$ multiply — $(D - i)$ multiplies.
- combining $D - i$ products into a running subtraction from $A_{i,d}$ — the expression $A_{i,d} - \sum$ accumulates $D - i$ subtractions.
- contributes for row $i$: $\mathrm{ML}\mathrel{+}= (D-i)$, $\quad\mathrm{AS}\mathrel{+}= (D-i)$.

**Divide** — $x_i = (\dots)/U_{i,i}$:
- $1$ divide.
- contributes for row $i$: $\mathrm{DV}\mathrel{+}= 1$.

There is no square root anywhere in a triangular solve.

### Per-column subtotals

Sum over rows $i = 1, \dots, D$. The quantity $D - i$ takes the values $D-1, D-2, \dots, 0$, so

$$
\sum_{i=1}^{D}(D - i) = \sum_{k=0}^{D-1} k = \frac{D(D-1)}{2}.
$$

**Adds + subtracts** (per column):

$$
\mathrm{AS}_{\text{col}} = \sum_{i=1}^{D}(D-i) = \frac{D(D-1)}{2}
$$

**Multiplies** (per column):

$$
\mathrm{ML}_{\text{col}} = \sum_{i=1}^{D}(D-i) = \frac{D(D-1)}{2}
$$

**Divides** (per column) — one per row:

$$
\mathrm{DV}_{\text{col}} = \sum_{i=1}^{D} 1 = D
$$

**Square roots** (per column):

$$
\mathrm{SQ}_{\text{col}} = 0
$$

## Per-matrix count

One batch element is a full solve $U \backslash A$ with a $D \times D$ RHS — that is $D$ right-hand-side columns. Each column is handled by one thread and costs the per-column tallies above. The per-matrix count is therefore $D$ times the per-column count:

**Adds + subtracts:**

$$
\boxed{\;\mathrm{AS}(D) = D \cdot \frac{D(D-1)}{2} = \frac{D^2(D-1)}{2}\;}
$$

**Multiplies:**

$$
\boxed{\;\mathrm{ML}(D) = D \cdot \frac{D(D-1)}{2} = \frac{D^2(D-1)}{2}\;}
$$

**Divides:**

$$
\boxed{\;\mathrm{DV}(D) = D \cdot D = D^2\;}
$$

**Square roots:**

$$
\boxed{\;\mathrm{SQ}(D) = 0\;}
$$

## Total

$$
\text{flops_trig_backsolve}(D) = \mathrm{AS}(D) + \mathrm{ML}(D) + \mathrm{DV}(D) + \mathrm{SQ}(D)
$$

$$
= \frac{D^2(D-1)}{2} + \frac{D^2(D-1)}{2} + D^2 + 0
= D^2(D-1) + D^2
$$

$$
\boxed{\;\text{flops_trig_backsolve}(D) = D^3\;}
$$

The two multiply-add tallies ($\tfrac{D^2(D-1)}{2}$ each) combine to $D^3 - D^2$, and the $D^2$ divides bring it to exactly $D^3$. The clean result is a coincidence of the algebra — the $-D^2$ from the multiply-add part and the $+D^2$ from the divides cancel.

## Validation

**Leading term.** A single triangular solve $U \backslash b$ for one vector is the textbook $D^2$ flops; with $D$ right-hand-side columns the cost is $D \cdot D^2 = D^3$. The derived total matches.

**What the split tells you.** Multiply and add each contribute $\tfrac{D^2(D-1)}{2} \sim \tfrac{1}{2}D^3$, so the kernel is overwhelmingly multiply-add work. Divides are only $O(D^2)$ — at $D=16$, $\mathrm{DV} = 256$ against $\mathrm{AS}+\mathrm{ML} = 3840$, about $6\%$ — and there are no square roots at all. So although divide is the expensive operation, it is a small minority of the count, and more so as $D$ grows.

**Spot checks** (per-type closed forms vs direct count):

| $D$ | $\mathrm{AS}=\mathrm{ML}=\tfrac{D^2(D-1)}{2}$ | $\mathrm{DV}=D^2$ | total $D^3$ |
|-----|-----|-----|-----|
| $2$ | $\tfrac{4\cdot1}{2}=2$ | $4$ | $2+2+4 = 8 = 2^3$ |
| $3$ | $\tfrac{9\cdot2}{2}=9$ | $9$ | $9+9+9 = 27 = 3^3$ |
| $4$ | $\tfrac{16\cdot3}{2}=24$ | $16$ | $24+24+16 = 64 = 4^3$ |

Direct check at $D=3$, one column: rows $i=3,2,1$ have $D-i = 0,1,2$ multiply-adds and $1$ divide each — per column $\mathrm{AS}=\mathrm{ML}=0+1+2=3$, $\mathrm{DV}=3$. Per matrix ($\times D = 3$): $\mathrm{AS}=\mathrm{ML}=9$, $\mathrm{DV}=9$. Matches the table.

## Result for `flops.jl`

$$
\text{flops_trig_backsolve}(D) = D^3
$$

Companion quantities (confirmed):

- $\text{dram_bytes_trig_backsolve}(D) = 3 D^2 \cdot 4$ — input $U$ dense $D\times D$ read, input $A$ dense $D\times D$ read, output $C$ dense $D\times D$ written; $3$ arrays.
- $\text{n_matrices_trig_backsolve}(D) = \lceil 10^9 / (4 \cdot 3 \cdot D^2) \rceil$.
