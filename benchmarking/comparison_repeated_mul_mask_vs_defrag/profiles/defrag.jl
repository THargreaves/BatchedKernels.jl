using Random
using LinearAlgebra

include("../../config/Schedule.jl")
include("../repeated_mul_defrag.jl")

Random.seed!(1234)

T = Float32
Dx = 4
Dy = 8
D = max(Dx, Dy)
N = Int(ceil(1e9 / (4 * 2 * Dx * Dy)))
# N = 2^9 + 1
n_muls = 20
nthreads = Schedule.best_nthreads("matmul", D)

M_in_cpu = rand(T, Dx, Dy, N)
A_cpu = rand(T, Dy, Dx)
M_out_cpu = zeros(T, Dx, Dx, N)
    
Ms_out = cu(M_out_cpu)
Ms_in = cu(M_in_cpu)
A = cu(A_cpu)

nblocks = cld(N, nthreads ÷ 32 * (32 ÷ D))

# Warm-up
for _ in 1:2
    CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_mul_defrag!(
        Ms_out, Ms_in, A,
        Val(Int32(Dx)), Val(Int32(Dy)), Val(Int32(D)),
        Val(Int32(nthreads)), Val(Int32(n_muls)), Int32(N),
    )
end

# Profile
CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_mul_defrag!(
    Ms_out, Ms_in, A,
    Val(Int32(Dx)), Val(Int32(Dy)), Val(Int32(D)),
    Val(Int32(nthreads)), Val(Int32(n_muls)), Int32(N),
)

# ncu --set full --target-processes all --import-source yes \
#     --kernel-name-base demangled \
#     --launch-skip 2 --launch-count 1 \
#     -o defrag_n_steps_20 -f \
#     julia --project=../../../. defrag.jl

# ncu --import defrag_n_steps_20.ncu-rep --page source --print-source sass --csv > defrag_n_steps_20_sass.csv

# ncu --import defrag_n_steps_20.ncu-rep --csv --page raw > defrag_n_steps_20.csv