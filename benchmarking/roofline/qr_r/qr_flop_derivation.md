# Flop count: Householder QR (R-only), `flops_qr_r(D)`

## Setup

Householder QR of a $D \times D$ matrix. Thread owns row $i$; the outer loop runs over columns $j = 1, \dots, D-1$. At step $j$ the reflector acts on the trailing block of rows $i \ge j$ and columns $t \ge j$.

Define the **active size** at step $j$:

$$m = D - j + 1$$

As $j$ runs $1 \to D-1$, the active size $m$ runs $D \to 2$.

## Counting conventions

- **Algorithmic flops.** Each multiply, add, subtract, divide, and `sqrt` counts as $1$ flop. They are *summed with equal weight* in the final formula — but they are **counted separately** below, because on real hardware their costs differ sharply: add/sub/multiply issue at full FP32 throughput, while divide and `sqrt` are multi-instruction sequences (roughly an order of magnitude slower). Splitting the tally makes visible *how much* of the work is cheap multiply-add versus expensive divide/sqrt — useful when interpreting achieved FLOP/s against the compute roof.
- **Reductions count as $m-1$ adds.** A `warp_reduce_sum` over $m$ lanes computes a sum of $m$ values, which is $m-1$ additions — regardless of whether done linearly or as a log-depth tree. The tree performs more *physical* adds ($\sim m\log_2 m$), but the *algorithm* requires $m-1$. The warp-reduction is an implementation detail.
- **Per-lane vs once-per-step.** When $m$ lanes each compute a *distinct* vector element, that is $m$ flops. When all lanes redundantly compute the *same broadcast scalar*, that is $1$ flop — the redundancy is implementation, like the reduction.

We track three tallies per step: $\mathrm{AS}$ = adds + subtracts, $\mathrm{ML}$ = multiplies, $\mathrm{DV}$ = divides, $\mathrm{SQ}$ = square roots.

## Per-step count

Walking the kernel body inside `if i >= j`, at a step with active size $m$.

**1. Column norm** — `norm_sq = warp_reduce_sum(R_col[j] * R_col[j])`
- $R_col[j]^2$: $m$ lanes square distinct elements — $m$ multiplies.
- reduction of $m$ values — $m-1$ adds.
- contributes: $\mathrm{ML}\mathrel{+}= m$, $\quad\mathrm{AS}\mathrel{+}= m-1$.

**2. Reflector entry** — `v_elem = R_col[j] - ifelse(i==j, -sign*sqrt(norm_sq), 0)`
- $\mathrm{sqrt}(\texttt{norm_sq})$ on the $i=j$ lane — $1$ square root.
- $-\,\texttt{sign}\cdot\mathrm{sqrt}(\dots)$ — $1$ multiply (by $\pm 1$).
- the subtraction modifies only the $j$-th entry of $v = (\text{column}) - \lVert\cdot\rVert e_j$ — $1$ subtract.
- contributes: $\mathrm{SQ}\mathrel{+}= 1$, $\quad\mathrm{ML}\mathrel{+}= 1$, $\quad\mathrm{AS}\mathrel{+}= 1$.

**3. Broadcast `v1`** — `shfl_sync` — data movement, $0$ flops.

**4. Normalise** — `v_elem /= v1`
- $m$ lanes each divide their own $v$-element — $m$ divides.
- contributes: $\mathrm{DV}\mathrel{+}= m$.

**5. Tau** — `tau = 2 / warp_reduce_sum(v_elem * v_elem)`
- $v_elem^2$: $m$ lanes, distinct — $m$ multiplies.
- reduction — $m-1$ adds.
- $2/(\dots)$ produces one scalar — $1$ divide.
- contributes: $\mathrm{ML}\mathrel{+}= m$, $\quad\mathrm{AS}\mathrel{+}= m-1$, $\quad\mathrm{DV}\mathrel{+}= 1$.

**6. Broadcast `tau`** — `shfl_sync` — $0$ flops.

**7. `tau_v_elem = tau * v_elem`**
- $m$ lanes, distinct products — $m$ multiplies.
- contributes: $\mathrm{ML}\mathrel{+}= m$.

**8. Trailing update** — the `t` loop runs $t = j, \dots, D$, i.e. $m$ columns. Per column $t$:
- `w_t = warp_reduce_sum(v_elem * R_col[t])` — $m$ multiplies $+\ (m-1)$ adds.
- `w_t = shfl_sync(...)` — $0$.
- `R_col[t] -= tau_v_elem * w_t` — $\texttt{tau_v_elem}\cdot w_t$ is $m$ distinct products ($m$ mul); the subtraction is $m$ subtracts.
- per column: $\mathrm{ML}\mathrel{+}= 2m$, $\quad\mathrm{AS}\mathrel{+}= (m-1) + m = 2m - 1$.
- over $m$ columns: $\mathrm{ML}\mathrel{+}= 2m^2$, $\quad\mathrm{AS}\mathrel{+}= m(2m-1) = 2m^2 - m$.

### Per-step subtotals

**Adds + subtracts** $\mathrm{AS}$, from parts 1, 2, 5, 8:

$$
\mathrm{AS}(m) = (m-1) + 1 + (m-1) + (2m^2 - m) = 2m^2 + m - 1
$$

**Multiplies** $\mathrm{ML}$, from parts 1, 2, 5, 7, 8:

$$
\mathrm{ML}(m) = m + 1 + m + m + 2m^2 = 2m^2 + 3m + 1
$$

**Divides** $\mathrm{DV}$, from parts 4, 5:

$$
\mathrm{DV}(m) = m + 1
$$

**Square roots** $\mathrm{SQ}$, from part 2:

$$
\mathrm{SQ}(m) = 1
$$

Cross-check: the total per step is $\mathrm{AS}+\mathrm{ML}+\mathrm{DV}+\mathrm{SQ} = (2m^2+m-1)+(2m^2+3m+1)+(m+1)+1 = 4m^2 + 5m + 2$, matching the combined count.

## Summing over all steps

The outer loop runs $j = 1, \dots, D-1$, so $m$ takes values $2, 3, \dots, D$. Use:

$$
\sum_{m=2}^{D} m^2 = \frac{D(D+1)(2D+1)}{6} - 1,
\qquad
\sum_{m=2}^{D} m = \frac{D(D+1)}{2} - 1,
\qquad
\sum_{m=2}^{D} 1 = D - 1.
$$

### Adds + subtracts

$$
\text{AS}(D) = \sum_{m=2}^{D}\left(2m^2 + m - 1\right)
= 2\!\left[\tfrac{D(D+1)(2D+1)}{6}-1\right] + \left[\tfrac{D(D+1)}{2}-1\right] - (D-1)
$$

$$
\boxed{\;\text{AS}(D) = \tfrac{1}{3}D(D+1)(2D+1) + \tfrac{1}{2}D(D+1) - D - 2\;}
$$

### Multiplies

$$
\text{ML}(D) = \sum_{m=2}^{D}\left(2m^2 + 3m + 1\right)
= 2\!\left[\tfrac{D(D+1)(2D+1)}{6}-1\right] + 3\!\left[\tfrac{D(D+1)}{2}-1\right] + (D-1)
$$

$$
\boxed{\;\text{ML}(D) = \tfrac{1}{3}D(D+1)(2D+1) + \tfrac{3}{2}D(D+1) + D - 6\;}
$$

### Divides

$$
\text{DV}(D) = \sum_{m=2}^{D}(m + 1)
= \left[\tfrac{D(D+1)}{2}-1\right] + (D-1)
$$

$$
\boxed{\;\text{DV}(D) = \tfrac{1}{2}D(D+1) + D - 2\;}
$$

### Square roots

$$
\boxed{\;\text{SQ}(D) = \sum_{m=2}^{D} 1 = D - 1\;}
$$

## Total

The algorithmic flop count sums the four tallies with equal weight:

$$
\text{flops_qr_r}(D) = \text{AS}(D) + \text{ML}(D) + \text{DV}(D) + \text{SQ}(D)
$$

$$
= \underbrace{\tfrac{2}{3}D(D+1)(2D+1)}_{\text{AS}+\text{ML cubic part}}
+ \underbrace{\big(\tfrac{1}{2}+\tfrac{3}{2}+\tfrac{1}{2}\big)D(D+1)}_{\text{quadratic}}
+ \underbrace{(-1+1+1+1)D}_{\text{linear}}
+ \underbrace{(-2-6-2-1)}_{\text{constant}}
$$

$$
\boxed{\;\text{flops_qr_r}(D) = \tfrac{2}{3}D(D+1)(2D+1) + \tfrac{5}{2}D(D+1) + 2D - 11\;}
$$

This is identical to the combined count — the split is just more telling: it shows the work is dominated by adds and multiplies (each $\sim\tfrac{1}{3}D(D+1)(2D+1)$, i.e. $\sim\tfrac{2}{3}D^3$ apiece), while divides are only $O(D^2)$ and square roots only $O(D)$. So although divide and `sqrt` are individually expensive on hardware, they are a vanishing fraction of the count at large $D$ — the kernel is overwhelmingly multiply-add work.

## Validation

**Leading term.** The cubic part is $\tfrac{2}{3}\cdot 2D^3 = \tfrac{4}{3}D^3$, matching textbook Householder QR, $\tfrac{4}{3}D^3 + O(D^2)$. The exact formula retains the $O(D^2)$ and $O(D)$ corrections, which matter at the small $D$ used here.

**Spot checks** (per-step $4m^2+5m+2$ summed vs closed form):

| $D$ | steps ($m$) | per-step sum | closed form | match |
|-----|-------------|--------------|-------------|-------|
| $2$ | $2$ | $28$ | $\tfrac{2}{3}(2)(3)(5)+\tfrac{5}{2}(2)(3)+4-11 = 20+15+4-11 = 28$ | OK |
| $3$ | $3,2$ | $53+28 = 81$ | $\tfrac{2}{3}(3)(4)(7)+\tfrac{5}{2}(3)(4)+6-11 = 56+30+6-11 = 81$ | OK |
| $4$ | $4,3,2$ | $86+53+28 = 167$ | $\tfrac{2}{3}(4)(5)(9)+\tfrac{5}{2}(4)(5)+8-11 = 120+50+8-11 = 167$ | OK |

**Per-type spot check at $D=3$** ($m=3$ then $m=2$):
- $\text{AS}$: $(2\cdot9+3-1)+(2\cdot4+2-1) = 20+9 = 29$; formula $\tfrac{1}{3}(3)(4)(7)+\tfrac{1}{2}(3)(4)-3-2 = 28+6-5 = 29$. OK.
- $\text{ML}$: $(2\cdot9+9+1)+(2\cdot4+6+1) = 28+15 = 43$; formula $\tfrac{1}{3}(3)(4)(7)+\tfrac{3}{2}(3)(4)+3-6 = 28+18-3 = 43$. OK.
- $\text{DV}$: $(3+1)+(2+1) = 7$; formula $\tfrac{1}{2}(3)(4)+3-2 = 6+1 = 7$. OK.
- $\text{SQ}$: $2$; formula $3-1 = 2$. OK.
- sum $29+43+7+2 = 81$. OK.

## Result for `flops.jl`

$$
\text{flops_qr_r}(D) = \tfrac{2}{3}D(D+1)(2D+1) + \tfrac{5}{2}D(D+1) + 2D - 11
$$

Companion quantities (confirmed):

- $\text{dram_bytes_qr_r}(D) = 2 D^2 \cdot 4$ — input $A$ dense $D\times D$ read, output $R$ transferred as a full dense $D\times D$, $2$ arrays.
- $\text{n_matrices_qr_r}(D) = \lceil 10^9 / (4 \cdot 2 \cdot D^2) \rceil$.
