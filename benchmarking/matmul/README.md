# Matmul intermediate-storage benchmarks

Controlled experiments on whether keeping computed intermediates in single-access
shared memory, instead of registers, helps a matmul-only graph:

```julia
X = A * B
Y = B * A
U = A * A
V = B * B
return ((U * V) * Y) * X
```

Only the residence of X (or X and Y) changes; inputs, operation variants,
orientations, schedule and block size are fixed within each comparison. Every
candidate is checked against a CPU reference.

**Float32 (RTX 4090, D32, batch 8193):** sharing X and Y cuts allocated registers
from 255 to 168 with no spills, but does not improve throughput (about 4% slower
than all-register at 64 threads). Reduced register pressure alone is not
sufficient for a speedup. Results: `intermediate_results.csv`.

```sh
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/matmul/intermediate_liveness.jl  # CPU order audit
OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/matmul/intermediate_storage.jl
```

`MATMUL_D=16`, `MATMUL_GRAPH=reused` and `MATMUL_ORDER=register_min` select the
other configurations.

**Float64:** `float64_storage.jl` compares all-register, X-shared and X/Y-shared
placements in both precisions. Local RTX 4090 resource checks are in
`float64_rtx4090_resources.csv`. Timing on an A100 uses the recording run script:

```sh
bash benchmarking/matmul/run_precision_benchmark.sh               # inside a GPU job
MODE=resources bash benchmarking/matmul/run_precision_benchmark.sh
```

`CASES` (e.g. `32:32,32:64`), `PRECISIONS`, `BATCH` and `RESULT_DIR` override the
defaults. No A100 matmul result data is archived in this repository.
