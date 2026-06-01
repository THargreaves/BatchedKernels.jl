# Shared-memory traffic: analytical cross-check (matmul, `ours`)

A worked validation that the measured shared-memory traffic
(`shmem_bytes_pm`, derived from the SASS `L1 Wavefronts Shared` counts)
matches a hand count of the kernel's shared accesses. Done once for
matmul as a methodology check.

## Counting convention

Count every shared-memory access in one matrix's worth of work: one
element read or written from shared memory = one access = 4 B (Float32).

**Broadcasting does not reduce the count.** When several threads read
the same shared element in lockstep, the access is *conflict-free* (it
costs no extra over a single access), but the bytes booked are the same
as if the reads were distinct — broadcasting optimises bank conflicts,
not bytes moved. So a shared element read by $D$ threads counts as $D$
accesses.

## Compute (`batch_op!(*, ...)`)

For $C = A B$, thread $d$ owns column $d$ of $C$;
$C[i,d] = \sum_{k} A[i,k]\,B_\text{col}[k]$, loops $i,k = 1..D$.

- **A:** thread $d$ reads every $A[i,k]$ → $D^2$ reads per thread; $D$
  threads per matrix → $D \cdot D^2 = D^3$ accesses. (All $D$ threads
  read the same element — broadcast — but per the convention that is
  still $D$ accesses.)
- **B:** thread $d$ reads column $d$ only, $D$ elements; $D$ threads,
  no sharing → $D^2$ accesses.
- **C:** each of $D^2$ elements written once → $D^2$ accesses.

$$
\text{compute}(D) = D^3 + 2D^2
$$

## Staging

Each operand makes one global$\leftrightarrow$shared round trip through
an intermediate layout. Per element, per operand:

- `intermediate_layout_load!`  global$\to$shared : 1 shared store
- `interm_to_dual_transfer!`   shared$\to$shared : 1 load + 1 store

= 3 shared accesses per element on the way in; the write-out path
(`dual_to_interm_transfer!` + `intermediate_layout_write!`) mirrors it
at 3 per element. Three operands (A, B in; C out), $D^2$ elements each:

$$
\text{staging}(D) = 3 \times 3 \times D^2 = 9D^2
$$

## Total

$$
\boxed{\;\text{shmem\_words\_per\_matrix}(D) = D^3 + 11D^2\;}
\qquad
\text{bytes} = 4\,(D^3 + 11D^2)
$$

## Validation

| $D$ | formula (words) | measured `shmem_bytes_pm` | measured (words) | ratio |
|-----|-----------------|---------------------------|------------------|-------|
| $2$ | $8 + 44 = 52$    | $208$ B                   | $52.0$           | 1.000 |
| $8$ | $512 + 704 = 1216$ | $5008$ B               | $1252.1$         | 1.030 |

$D=2$ is exact. $D=8$ matches to 3%; the small excess is boundary
handling in the layout-transfer loops (a few predicated `STS`, visible
in the SASS as `@!P0`/`@!P1` instructions) that the idealised "3 per
element" omits.

## Validity for $D$ not dividing 32

A warp packs $32/D$ matrices. The SASS `L1 Wavefronts Shared` counter
charges one whole wavefront (128 B) per shared instruction **regardless
of how many lanes are active** — it counts wavefronts, not active
lanes. So:

- **$D \mid 32$** (here $D \in \{2,4,8,16\}$): the warp is fully packed,
  every lane active, no masking. The formula is exact and equals the
  SASS figure.
- **$D \nmid 32$** ($D = 6,10,12,14$): $32 \bmod D$ lanes are masked off
  (the kernel's `dual_padding`). Those instructions still issue full
  wavefronts, so the SASS figure carries a small masking overhead the
  idealised count omits — the formula slightly *under*-reads the SASS
  there.

All benchmarked dimensions are powers of two, so the formula is exact at
every measured point.
