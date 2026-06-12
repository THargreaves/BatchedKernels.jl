§ Dual-access vs naive layout: ablation and trade-off characterization
[Opening — set up the hypothesis to test.]
The dual-access layout's design goal is bank-conflict immunity through padding. The naive layout (column-major, no padding) is the natural alternative. We ablate the two layouts on the Kalman kernel across a range of D values to validate the design choice.
[Method — short paragraph.]
We profile both layouts at D ∈ {8, 10, 16, 17, 18, 27} (chosen to span the relevant arithmetic regimes: powers of 2, odd values, non-power-of-2 even values). For each, we collect SASS-level bank-conflict wavefront counts, instruction-mix breakdown (including wide-load opcodes LDS.128 / LDS.64), runtime, and occupancy-relevant metrics (shared memory, registers, eligible warps per cycle).
[Result 1 — the gcd-of-D-and-32 mechanism for bank conflicts.]
[Table or plot of excess_fraction vs D for naive (dual is always zero).]
Bank-conflict severity in naive scales with gcd(D, 32). At D=16 (gcd=16), column-stride accesses produce 16-way conflicts, giving 207% wavefront overhead. At gcd=2 D values (10, 18), conflicts are mild (12–20%). At odd D (17, 27, gcd=1), no conflicts. Dual-access padding eliminates conflicts at all D.
[Result 2 — the LDS.128 vectorization mechanism.]
[Table or plot of wide_load_float_fraction vs D.]
Naive's column-major layout permits the compiler to emit LDS.128 instructions (16-byte vector shared-memory loads, four floats per instruction) on stride-1 accesses. Approximately 14–23% of naive's float loads use wide instructions at even D, where 4-element-aligned chunks exist. At odd D the compiler cannot find 4-aligned access patterns and falls back to scalar loads. Dual-access's transformed layout breaks the contiguity required by LDS.128: zero wide loads at all D.
[Result 3 — the trade-off.]
[The headline plot: dual/naive time ratio vs D.]
Whether dual-access wins or loses depends on the balance between bank-conflict savings (dual's advantage) and LDS.128 vectorization (naive's advantage):

At gcd(D,32) ≥ 8 (powers of 2): naive's wavefront overhead is catastrophic (>200% at D=16). Dual-access wins decisively (28% faster at D=16). Vectorization cannot compensate.
At odd D: neither layout has conflicts, neither uses wide loads. Tied at D=17. Dual wins at D=27 — see Result 4.
At non-power-of-2 even D (D=10, 18): mild conflicts (12–20% wavefront overhead) compete with ~15–20% LDS.128 wide-load coverage. At D=18 vectorization wins (naive 10% faster); at D=10 conflicts win (dual faster).

[Result 4 — the third mechanism: shared-memory pressure and occupancy.]
At D=27, naive uses 97.6 KB of shared memory per block — within 3% of the RTX 4090's 100 KB opt-in maximum. SM throughput collapses to 78% (vs 98% elsewhere), eligible warps per cycle drop to 0.29 (vs 0.4+ elsewhere). The occupancy bottleneck dominates regardless of bank conflicts and vectorization. Dual-access at D=27 [either uses less shared memory, leaving headroom for higher occupancy, OR shows similar collapse — to be confirmed when orig_27 is profiled].
[Synthesis — the design argument.]
[Plot: ns/inst vs D for both layouts.]
Dual-access exhibits near-constant per-instruction time (2.5–2.9 ns/inst) across all D. Naive is bimodal: efficient at favorable D (2.7–2.9 ns/inst at D=10, 18), inefficient at unfavorable D (4.0–4.1 ns/inst at D=16, 27).
For an auto-generator emitting kernels at user-specified D, dual-access is the robust default: its worst case (~10% slower than naive at D=18) is bounded; naive's worst cases (3× wavefront overhead at D=16; occupancy collapse at high D) are catastrophic. A D-aware layout-selection heuristic could pick naive at the narrow band where it wins; this is left as future work.