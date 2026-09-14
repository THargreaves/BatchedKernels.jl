M4 validates compute bodies and their host registry before integration into the
fuser. Run the test items using the temporary-environment commands in
[`../memory/README.md`](../memory/README.md), replacing `test/memory` with
`test/variants`. Run correctness with `debug_accessors` both false and true;
production register cases also require zero compiled local memory.

The hybrid registry covers Float32/Float64 matmul, Float32 addition/subtraction,
out-of-place upper Cholesky, and out-of-place triangular solves. Logical dimensions
must fit a complete group of at most 32 lanes. Unsupported requests return no
hybrid variants and retain the existing legacy path and its existing restrictions.
Float64 named hybrid variants are currently limited to matmul; the forced-assignment
path validates operation support and byte accounting for the selected element type.

Contracts relevant to future code generation:

- Every lane in a complete matrix group participates, including padded lanes that
  do not own an output line. Callers synchronize shared producers before the body
  and shared consumers or storage reuse after it.
- Cholesky and solve use fresh private scratch even with shared input/output. Their
  internal recurrences use registers/shuffles; this avoids shared recurrence fences
  but still consumes registers. Resource use must be checked on full fused kernels.
- Matmul snapshots one owned line of its right operand before accumulating output,
  preventing repeated shared loads. Mirrored calls swap/adjoint operands.
- Alias candidate positions do not authorize reuse by themselves. Every overlapping
  input must have the same pointwise element-to-lane/address map; `A + A'` fails that
  condition. M5 must validate owners through wrappers. Registers never alias tape
  operands in this version; forced mutation retains the existing shared path.
- Tests keep register-view construction and use in one participation branch. On the
  current compiler, splitting these across reconverged guards prevented allocation
  elimination. This is an observed code-generation constraint, not an API ban on
  branches. The factorization view factory also required explicit callsite inlining
  to eliminate allocation escape. Ordinary compute-variant calls passed. Final
  inference and binary resource checks remain necessary.

Selected shapes exercise rectangular source participation, non-power-of-two groups,
full-warp masks, triangular masking and different residences without expanding a
Cartesian product. CPU and legacy-dual numerical references are used where the
legacy operation supports that case. Microkernel resource acceptance does not
establish fused throughput; that is the M6 decision gate.

The row solve uses an explicit GPU compiler constraint consuming each solved scalar
in every participating lane. Without it, the tested N=32, P=2 specialization grouped
528 factor broadcasts before the arithmetic and spilled despite scalarized PTX.
The empty assembly adds no GPU synchronization or memory fence; it constrains dead
lane optimization and must remain accompanied by final binary resource checks.
The N=32, P=2 register case is a zero-local-memory regression test, without pinning
an exact register count to one compiler version.

The separate column solve requires logical ColAccess for the factor, RHS and output.
It broadcasts N×P solved RHS values and holds P private RHS entries per lane. Tests
cover P>N, padded groups and unit diagonals through transposing wrappers (with NaN
stored diagonals). Lower/upper and unit/nonunit factors remain supported. An outer
transpose/adjoint flips the physical convention required of its parent.

Both solve variants remain candidates. In particular, using column solves for both
U′ and U requires opposite physical orientations of U; storage or conversion costs
must be included in fused selection. No dimension or residence is globally disabled,
and the compiler constraint is not a guarantee against spilling in a larger kernel.
