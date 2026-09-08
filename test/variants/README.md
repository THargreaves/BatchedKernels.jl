M4 validates compute bodies and their host registry before integration into the
fuser. Run the test items using the temporary-environment commands in
[`../memory/README.md`](../memory/README.md), replacing `test/memory` with
`test/variants`. Run correctness with `debug_accessors` both false and true;
production register cases also require zero compiled local memory.

The hybrid registry currently covers Float32 matmul, addition/subtraction,
out-of-place upper Cholesky, and out-of-place triangular solves. Logical dimensions
must fit a complete group of at most 32 lanes. Unsupported requests return no
hybrid variants and retain the existing legacy path and its existing restrictions.
The registry does not yet drive public fused execution.

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
to eliminate allocation escape. Ordinary compute-variant calls passed. Final inference and binary resource checks remain necessary.

Selected shapes exercise rectangular source participation, non-power-of-two groups,
full-warp masks, triangular masking and different residences without expanding a
Cartesian product. CPU and legacy-dual numerical references are used where the
legacy operation supports that case. Microkernel resource acceptance does not
establish fused throughput; that is the M6 decision gate.

A tested N=32, P=2 register solve reached the hardware register limit and spilled
during assembly despite fully scalarized PTX (255 registers, 1040–1048 local bytes
for the two factor orientations on the tested RTX 4090 toolchain). The selected
single-shared alternative uses 56 registers and zero local memory. Such specializations remain ineligible
under the zero-local-memory gate; support for their shape does not promise every
residence/orientation choice is feasible. M6 must select a measured feasible
assignment. Adding per-step warp barriers made this case worse and was discarded.
