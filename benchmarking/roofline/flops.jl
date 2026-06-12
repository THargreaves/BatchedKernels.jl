BYTES_PER_ELEM = 4.0

flops_matmul(D::Integer)      = 2.0 * D^3 - D^2
dram_bytes_matmul(D::Integer) = 3.0 * D^2 * BYTES_PER_ELEM
batch_size_matmul(D::Integer) = ceil(Int, 1e9 / (4 * 3 * D^2))

flops_cholesky(D::Integer)      = (D^3 - D) / 3 + D * (D - 1) / 2 + D
dram_bytes_cholesky(D::Integer) = 2.0 * D^2 * BYTES_PER_ELEM
batch_size_cholesky(D::Integer) = ceil(Int, 1e9 / (4 * 2 * D^2))

flops_trig_backsolve(D::Integer)      = Float64(D)^3
dram_bytes_trig_backsolve(D::Integer) = 3.0 * D^2 * BYTES_PER_ELEM
batch_size_trig_backsolve(D::Integer) = ceil(Int, 1e9 / (4 * 3 * D^2))

flops_qr_r(D::Integer)      = (2/3)*D*(D+1)*(2*D+1) + (5/2)*D*(D+1) + 2*D - 11
dram_bytes_qr_r(D::Integer) = 2.0 * D^2 * BYTES_PER_ELEM
batch_size_qr_r(D::Integer) = ceil(Int, 1e9 / (4 * 2 * D^2))

flops_matadd(D::Integer) = Float64(D)^2
flops_matsub(D::Integer) = Float64(D)^2

flops_trmm(D::Integer) = Float64(D)^3

flops_qr_r_tall(D1::Integer, D2::Integer) =
    2*D1*D2^2 + 8*D1*D2 - (2/3)*D2^3 - (7/2)*D2^2 + (31/6)*D2

flops_kalman(D::Integer) =
    6 * flops_matmul(D) +
    2 * flops_matadd(D) +
    flops_cholesky(D) +
    2 * flops_trig_backsolve(D) +
    flops_matsub(D)

function dram_bytes_kalman(D::Integer)
    N        = batch_size_kalman(D)
    nthreads = 256
    per_blk  = (nthreads ÷ 32) * (32 ÷ D)      # matrices per block
    nblocks  = cld(N, per_blk)
    elems    = (2.0 * D^2 * N + 4.0 * D^2 * nblocks) / N
    return elems * BYTES_PER_ELEM
end

batch_size_kalman(D::Integer) = ceil(Int, 1e9 / (4 * 2 * D^2))

flops_sqrt_kalman(D::Integer) =
    2 * flops_trmm(D) +
    flops_qr_r_tall(2*D, D) +
    flops_qr_r(2*D)

dram_bytes_sqrt_kalman(D::Integer) = dram_bytes_kalman(D)
batch_size_sqrt_kalman(D::Integer) = batch_size_kalman(D)

# Gaussian likelihood flops:
#     delta = x - mu             vector subtract       -> D
#     cholesky(Sigma)            D x D Cholesky        -> flops_cholesky(D)
#     log det Sigma = 2*sum log U[i,i]                 -> 2D
#         (D log + (D-1) add + 1 mul)
#     y = L \ delta              triangular vec solve  -> D^2
#         (D(D-1) mul+sub + D div)
#     mahal = |y|^2              squared norm          -> 2D - 1
#     combine -0.5*(D*log2pi + log_det + mahal)        -> 4
flops_gauss_likelihood(D::Integer) =
    flops_cholesky(D) + Float64(D)^2 + 5 * D + 3

dram_bytes_gauss_likelihood(D::Integer) =
    (1.0 * D^2 + 2.0 * D + 1.0) * BYTES_PER_ELEM

batch_size_gauss_likelihood(D::Integer) = ceil(Int, 1e9 / (4 * 1 * D^2))

const OP_FLOPS = Dict{String,Function}(
    "matmul" => flops_matmul, "cholesky" => flops_cholesky,
    "qr_r" => flops_qr_r, "trig_backsolve" => flops_trig_backsolve,
    "kalman" => flops_kalman, "sqrt_kalman" => flops_sqrt_kalman,
    "gauss_likelihood" => flops_gauss_likelihood,
)
const OP_DRAM_BYTES = Dict{String,Function}(
    "matmul" => dram_bytes_matmul, "cholesky" => dram_bytes_cholesky,
    "qr_r" => dram_bytes_qr_r, "trig_backsolve" => dram_bytes_trig_backsolve,
    "kalman" => dram_bytes_kalman, "sqrt_kalman" => dram_bytes_sqrt_kalman,
    "gauss_likelihood" => dram_bytes_gauss_likelihood,
)
const OP_BATCH_SIZE = Dict{String,Function}(
    "matmul" => batch_size_matmul, "cholesky" => batch_size_cholesky,
    "qr_r" => batch_size_qr_r, "trig_backsolve" => batch_size_trig_backsolve,
    "kalman" => batch_size_kalman, "sqrt_kalman" => batch_size_sqrt_kalman,
    "gauss_likelihood" => batch_size_gauss_likelihood,
)

"Per-matrix algorithmic flops for `op` at dimension `D`."
flops_per_batch_elem(op::AbstractString, D::Integer) = OP_FLOPS[op](D)

"Per-matrix analytic DRAM bytes for `op` at `D` (single-pass, Float32)."
dram_bytes_per_batch_elem(op::AbstractString, D::Integer) = OP_DRAM_BYTES[op](D)

"Batch size N used in the benchmark/profile runs for `op` at `D`."
batch_size(op::AbstractString, D::Integer) = OP_BATCH_SIZE[op](D)