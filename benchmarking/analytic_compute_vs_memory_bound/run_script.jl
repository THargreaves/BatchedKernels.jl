using Plots

# RTX 4090, FP32
const COMPUTE   = 82.6e12      # FLOPs/s
const BANDWIDTH = 1.008e12     # bytes/s
const B         = 4            # bytes/element

# Crossover D from  AI = ridge  ⟺  2D/(3b) = Compute/BW
const D_CROSS = 3 * B * COMPUTE / (2 * BANDWIDTH)

# Per-matmul time, in microseconds
t_mem(D)  = 3 * D^2 * B / BANDWIDTH * 1e6
t_comp(D) = 2 * D^3     / COMPUTE   * 1e6

D_max  = ceil(Int, 1.25 * D_CROSS)
Ds     = 1:D_max
tcross = t_mem(D_CROSS)

plot(Ds, t_mem.(Ds);
    label      = "Memory bound",
    lw         = 2,
    xlabel     = "D",
    ylabel     = "Time per matmul (μs)",
    legend     = :topleft,
    framestyle = :box,
    size = (400, 300),
    )

plt = plot!(Ds, t_comp.(Ds);
    label = "Compute bound",
    lw    = 2)

vline!([D_CROSS]; label = "", color = :gray, ls = :dash, alpha = 0.5)

scatter!([D_CROSS], [tcross];
    label = "Crossover",
    color = :black,
    ms    = 6)

annotate!(D_CROSS, tcross * 1.7,
    text("D ≈ $(round(Int, D_CROSS))",
         :center, 9))

path = joinpath(@__DIR__, "matmul_compute_vs_memory.png")
savefig(path)