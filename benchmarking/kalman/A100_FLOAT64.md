# Four-slot Kalman precision experiment

Run from the repository root inside an allocated A100 job:

```sh
julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
bash benchmarking/kalman/run_precision_benchmark.sh
```

Dependency installation can be done on the login/setup node beforehand. The script
uses the site's existing Julia/CUDA environment and does not submit a scheduler job.
Julia 1.12.6 matches local validation. Return the generated directory under
`benchmarking/kalman/precision_runs/`, including results.csv and run.log.

Defaults: Float32 and Float64, D32, 32 threads/block, batch 8193. Set MODE=resources
for batch-5 correctness/resource checks without timing. BATCH, PRECISIONS,
POLICIES, RESULT_DIR and JULIA_BIN are optional overrides. A smaller control is CASES=16:32.

The covariance update is the same as hybrid_m6.jl. P and H are batched; A, Q and R
are block-common inputs. The runner asserts that the legacy schedule requires four
matrix slots. H is repeated across batch members in these controlled inputs but
is represented and loaded as a batched input. This experiment concerns the storage
of intermediates, not the statistical variability of measurement matrices.

Policies retain all computed values in registers, put the predicted covariance
in single-access shared storage, or put the Cholesky factor in single-access shared
storage. A fourth policy shares both H and the predicted covariance. All use identical compute variants, orientations and column output staging.
An additional register_row_stage control uses row-oriented output staging, which
performed well in earlier Kalman tests. Compare shared candidates against the
faster of the two register baselines before claiming a throughput benefit.
The register baselines still use shared storage for common inputs and global I/O.
No register caps are applied. Local-memory candidates are retained and timed.

The focused D32 comparison uses one warp/block because the current static shared
allocation path has a per-block limit: common Float64 inputs plus shared scratch
can exceed it at larger block sizes. This is an existing launch-path constraint,
not an A100 hardware limit or a newly imposed framework restriction. Results do
not establish the best possible block size or placement.

Float64 Cholesky and triangular solve bodies now use scalar-type-specialized
scratch and constants. The row solve uses the double-precision equivalent of its
existing scalar-use compiler constraint (empty side-effecting assembly); no new
barriers or scheduling strategy are added. Elementwise variants also accept
matching Float64 operands. Legacy implementations are unchanged.

Every batch member is checked against a CPU covariance update (Float64 rtol=1e-11,
atol=1e-12; Float32 rtol=3e-4, atol=3e-5), and all inputs must remain unchanged.
Timing uses nine randomized rounds of twenty GPU-event-timed launches after warmup.
The CSV records median and quartiles, compiled register/local/shared allocation,
and theoretical occupancy. Local allocation is not dynamic spill traffic, and
occupancy is not measured achieved occupancy. Detailed bank-conflict profiling
remains separate. Repeat timings before drawing thesis conclusions.

## Local validation

On RTX 4090, D32/batch5, all four matched-column-staging policies pass the CPU
reference and input-preservation checks in both precisions. Float64 compiled local
bytes/thread are 832 (register), 496 (predicted shared), 744 (factor shared), and
216 (H and predicted shared). All four use 255 registers/thread. Thus sharing
reduces local allocation but does not eliminate it in this kernel. These are
resource observations, not A100 speed predictions.

The row-staging register control also passes both precisions. In Float64 it uses
255 registers and 712 local bytes/thread (versus 832 with column staging).
Exact resource observations for all ten candidates are in
[float64_rtx4090_resources.csv](float64_rtx4090_resources.csv).

Focused production regression checks pass (99 assertions across the registry,
matmul precision, and factorization tests); the factorization suite also passes
39 debug-accessor checks. Only three focused Float64 factorization cases were
added, covering Cholesky, row solve with an adjointed factor, and column solve,
including partial subgroup participation.

The launcher and timing/CSV path also pass a D16/Float32/batch5 smoke test across
all five policies. Those tiny-batch timings are not performance evidence.

## Targeted follow-up

The [mixed-storage investigation](MIXED_STORAGE.md) adds correction/output storage
candidates and a focused launcher which also tests 64-thread blocks where feasible:

```sh
bash benchmarking/kalman/run_mixed_storage_candidates.sh
```

The original launcher retains its five-policy, 32-thread defaults. Explicit
CASES=32:64 is now accepted; oversized static allocations produce a documented
SKIP record instead of attempting compilation.

The static-capacity gap is now addressed by the opt-in
[dynamic-shared comparison](DYNAMIC_SHARED.md). Run its dedicated launcher to
compare the existing register and H/predicted-shared placements at 32/64/128 threads
with identical dynamic allocation support for both strategies.
