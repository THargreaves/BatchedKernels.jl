# Kalman benchmarks

Scripts and archived results behind the Kalman performance claims. Timings are
median CUDA-event kernel times unless stated; every runner checks results against
a CPU reference before timing and records registers, local and shared bytes.
Run from the repository root.

## Generated filtering kernels (automatic policy)

RTX 4090, Julia 1.12.6, batch 8192, 128 threads. "Common" means the model inputs
are shared across the batch; the Float64 case has a batched transition matrix.

| State / obs | Type | Joseph step (auto / legacy) | SRKF step | Backward step |
| --- | --- | ---: | ---: | ---: |
| 3 / 2 | Float32 | 8.19 / 7.87 µs | 9.18 µs | 10.24 µs |
| 8 / 4 | Float32 | 14.51 / 16.38 µs | 17.89 µs | 21.28 µs |
| 16 / 6 | Float32 | 53.23 / 70.59 µs | 70.66 µs | 115.97 µs |
| 9 / 4 | Float64 | 190.78 µs / — | 379.06 µs | 473.09 µs |

The legacy planner cannot compile the Float64 Cholesky. Each CSV also includes a
forced all-shared candidate with the same operation bodies, first-call latency
and cached-call time.

```sh
julia --project benchmarking/kalman/automatic_joseph.jl    # automatic_joseph_results.csv
julia --project benchmarking/kalman/automatic_srkf.jl      # automatic_srkf_results.csv
julia --project benchmarking/kalman/automatic_backward.jl  # automatic_backward_results.csv
```

`sanitize_automatic.jl`, `sanitize_srkf.jl` and `sanitize_backward.jl` are small
correctness drivers for `compute-sanitizer --tool {memcheck,racecheck,synccheck}`;
the invocation is in each file's header.

## Single-access shared intermediates (Float64 D32, A100)

Covariance update `covariance_four` in `float64_storage.jl`, batch 8193; P and H
batched, A, Q and R common. The mixed policy keeps H and the predicted covariance
in single-access shared memory; the register policy keeps computed values in
registers. Both are measured with and without reuse of the A/Q/R common-input
buffers, whose consumer lifetimes do not overlap.

| Threads | Register | Register + reuse | Mixed | Mixed + reuse |
| --- | ---: | ---: | ---: | ---: |
| 32 | 3614.1 µs | 3107.2 µs | 3295.7 µs | 1746.6 µs |
| 64 | 2989.8 µs | 2975.4 µs | 2537.7 µs | 1748.0 µs |
| 128 | 2844.5 µs | 2994.7 µs | 2540.1 µs | 1470.0 µs |

Best mixed against best register configuration: **1.935× throughput**. Both winners
have 12.5% theoretical occupancy, so the gain is not an occupancy increase over
the register baseline. The mixed assignments are hand-selected, and common-input
reuse is a benchmark-only code-generation transformation (`common_reuse.jl`); the
automatic policy does not discover this kernel.

```sh
bash benchmarking/kalman/run_common_reuse_comparison.sh   # inside an A100 job
```

`common_reuse_a100_reported.csv` holds all 12 rows as reported from that run.
The run followed commit `ac9905e`, but its revision, device/MIG metadata and full
log were not archived, so treat it as unverified provenance until it is repeated.
The run scripts now record that metadata automatically.

Supporting RTX 4090 data from the same runners: `common_reuse_rtx4090.csv`,
`dynamic_shared_rtx4090_resources.csv` (`run_dynamic_shared_comparison.sh`),
`float64_rtx4090_resources.csv` (`run_precision_benchmark.sh`) and
`mixed_storage_*.csv` (`run_mixed_storage_candidates.sh`).

## Register intermediates versus the legacy planner (Float32, RTX 4090)

Covariance update from `hybrid_m6.jl`: fastest admitted legacy (all dual-access
shared) against register-heavy configurations.

| D | Batch | Legacy | Register | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 8 | 8192 | 13.11 µs | 10.19 µs | 1.29× |
| 16 | 8192 | 70.66 µs | 51.15 µs | 1.38× |
| 32 | 8192 | 802.41 µs | 374.32 µs | 2.14× |
| 32 | 131072 | 12.41 ms | 5.85 ms | 2.12× |

```sh
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/hybrid_m6.jl            # m6_results.csv
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/hybrid_m6_streaming.jl  # 1 GiB batch row
```

With all five inputs batched at D32, selective shared H with column staging took
490.34 µs against 657.45 µs for the best spill-free register baseline (1.34×).
This combines a storage and a staging change, so it is not a pure residence
ablation. See `selective_results.csv` and `selective_admission.csv`; reproduce with
`BK_SELECTIVE_FINALISTS=true julia --project=. benchmarking/kalman/hybrid_selective.jl`.
