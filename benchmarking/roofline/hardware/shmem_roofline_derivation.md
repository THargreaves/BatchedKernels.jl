# Shared-memory throughput — theory vs measurement (RTX 4090, AD102)

## Spec values

| Quantity | Value | Source |
|---|---|---|
| SM count (RTX 4090) | 128 | AD102, 16 of 144 SMs disabled |
| GPU boost clock | 2.52 GHz | NVIDIA Ada whitepaper |
| Shared-memory banks per SM | 32 | NVIDIA architecture |
| Bank width | 4 B | NVIDIA architecture |
| FP32 peak | 82.58 TFLOP/s | flat roofline ceiling |

## Theoretical upper bound

One conflict-free 32-bit shared access by a warp is one **wavefront**, moving

$$ 32 \text{ lanes} \times 4 \text{ B} = 128 \text{ B}. $$

The simple model of one wavefront per cycle per SM gives

$$
B_\text{theory}
= 128\ \text{B} \times 128\ \text{SM} \times 2.52\times10^{9}\ \text{Hz}
\approx 41.3\ \text{TB/s}.
$$

This is an upper bound, not an achievable rate: it assumes the shared-memory
data pipe is fed a new wavefront every cycle.

## Measured

The microbenchmark (`measure_shmem_bw.jl`: scalar `LDS`, conflict-free)
sustains **~18.3 TB/s**, about 44% of the 41.3 TB/s upper bound.

## Cause of the gap — two different pipeline stages

A scalar `LDS` passes through two stages, and the profiler reports each with a
separate metric. Both can be read from the per-kernel `ncu` raw CSV
(`ncu -i <report>.ncu-rep --csv --page raw`):

**Stage 1 — instruction issue.**
Column: `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active`.
Measured: **~100%**. This is the fraction of cycles the LSU *issue port* is
dispatching an instruction. At 100% the warp scheduler is launching an LSU
instruction every cycle it can.

**Stage 2 — shared-memory data pipe.**
Column: `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed`
(the counters `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed`
and `l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_elapsed` agree).
Measured: **~50%**. This is how fast the shared-memory data pipe — the stage
that actually routes lane addresses to banks and moves the bytes — processes
wavefronts, as a fraction of its rated peak.

**What the split means.** Issue at 100% but data pipe at 50% means a stream of
scalar 32-bit `LDS` cannot drive the shared-memory data pipe past half its
rated peak: either the data pipe accepts a wavefront only every other cycle,
or the issue slots are shared with the loop's address-update instructions so
only half of what is issued is an actual `LDS`. The metrics do not separate
these, but the conclusion is the same and is a measured fact: scalar-`LDS`
traffic tops out at ~50% of the data pipe's peak. Reaching the 41.3 TB/s
upper bound would require wider (vectorised) accesses — `LDS.64`/`LDS.128` —
that move more bytes per issued instruction.


## Which number to use for the roofline ceiling

Use the **measured ~18.3 TB/s**. Reasons:

- It is an empirical, achievable rate, measured under the same timing basis
  (BenchmarkTools median) as the roofline's y-axis.
- `ours` uses scalar `LDS`/`STS` exclusively (confirmed from SASS — no
  `LDS.64`/`LDS.128`), so it is subject to the same issue-bound limit as the
  microbenchmark. The scalar-load measured rate is therefore the correct,
  like-for-like ceiling for comparing against `ours`.
- Cross-check: `ours` at D=8 sustains ~16.6 TB/s of shared traffic — 91% of
  the 18.3 TB/s ceiling — independently confirming both the ceiling and the
  kernel's shared-memory-bound regime.

The 41.3 TB/s upper bound is reported only for context; it is not reachable by
a scalar-`LDS` kernel and is not used as the ceiling.
