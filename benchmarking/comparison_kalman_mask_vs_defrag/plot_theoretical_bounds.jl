using Plots

# FLOPs per op. Multiply-add counted as two FLOPs throughout.
# Rectangular generalisations of the square forms from the roofline
# analysis. Cholesky stays square-only (it acts on a Dy × Dy block in
# the Kalman kernel and has no rectangular counterpart here).
flops_matmul(m, k, n)      = 2.0 * m * k * n - m * n
flops_cholesky(D::Integer) = (D^3 - D) / 3 + D * (D - 1) / 2 + D
flops_trig_backsolve(m, n) = Float64(m)^2 * n      # m × m triangle, n RHS cols
flops_matadd(m, n)         = Float64(m) * n
flops_matsub(m, n)         = Float64(m) * n

n_mats_per_warp(D::Integer) = 32 ÷ D

"""Naive bound: floor(32/min(Dx,Dy)) / floor(32/max(Dx,Dy)).

This is the report's single-stage formula applied as if all of
Kalman's work were in the smaller dimension's space. Symmetric in
(Dx, Dy) and a strict upper bound."""
function naive_bound(Dx, Dy)
    return n_mats_per_warp(min(Dx, Dy)) / n_mats_per_warp(max(Dx, Dy))
end

"""Per-stage FLOPs for the Kalman defrag kernel.

Returns `(F_xx, F_yy)`: the total FLOPs across the 8 Dx-stage ops
and the 3 Dy-stage ops respectively. Per-op shapes are taken from
inspection of `kalman_defrag.jl`."""
function kalman_flops(Dx, Dy)
    # Dx-stage ops (run at Dx-packing in defrag).
    #   1, 2:  F·P, M3·F'                  matmul(Dx, Dx, Dx)
    #   3:     M1 + Q                       matadd(Dx, Dx)
    #   4:     HP = H · P_pred (or its transpose form, FLOP-symmetric).
    #                                       matmul(Dy, Dx, Dx)
    #   8, 9:  U' \ HP, U \ (U' \ HP)        trig_backsolve(Dy, Dx)
    #   10:    K_trans' · H                  matmul(Dx, Dy, Dx)
    #   11:    (I − KH) · P_pred             matmul(Dx, Dx, Dx)
    F_xx = 2 * flops_matmul(Dx, Dx, Dx) +
           flops_matadd(Dx, Dx) +
           flops_matmul(Dy, Dx, Dx) +
           2 * flops_trig_backsolve(Dy, Dx) +
           flops_matmul(Dx, Dy, Dx) +
           flops_matmul(Dx, Dx, Dx)

    # Dy-stage ops (run at Dy-packing in defrag).
    #   5: HP · H'                           matmul(Dy, Dx, Dy)
    #   6: S = HPH' + R                      matadd(Dy, Dy)
    #   7: Cholesky(S)                       cholesky(Dy)
    F_yy = flops_matmul(Dy, Dx, Dy) +
           flops_matadd(Dy, Dy) +
           flops_cholesky(Dy)

    return F_xx, F_yy
end

"""Kalman-weighted upper bound on the mask-vs-defrag runtime ratio.

Per-op wall time is (per-thread work) / M_packing, where M_packing =
⌊32/D_padded⌋ and per-thread work scales as F_i / n_i (FLOPs per
thread). For Dx-stage ops n_i = Dx, for Dy-stage ops n_i = Dy.

In mask, D_padded = max(Dx, Dy) for every op, so M_mask =
⌊32/max(Dx,Dy)⌋. In defrag, D_padded = n_i, so M_defrag = ⌊32/Dx⌋
for Dx-stage ops and ⌊32/Dy⌋ for Dy-stage ops. Summing over all 11
ops:

    T_mask   ∝ (1/M_mask) × (F_xx/Dx + F_yy/Dy)
    T_defrag ∝ F_xx / (Dx × ⌊32/Dx⌋) + F_yy / (Dy × ⌊32/Dy⌋)

so the ratio is

                  (1/M_mask) × (F_xx/Dx + F_yy/Dy)
    -------------------------------------------------------------
    F_xx / (Dx × ⌊32/Dx⌋) + F_yy / (Dy × ⌊32/Dy⌋)

The floors matter when Dx, Dy, or max(Dx,Dy) do not divide 32:
⌊32/D⌋·D < 32 in that case, so lanes are wasted. When all three
divide 32 the floors collapse and the expression reduces to
`max(Dx,Dy) × (F_xx/Dx + F_yy/Dy) / (F_xx + F_yy)`."""
function kalman_bound(Dx, Dy)
    F_xx, F_yy = kalman_flops(Dx, Dy)
    M_mask = n_mats_per_warp(max(Dx, Dy))
    M_xx = n_mats_per_warp(Dx)
    M_yy = n_mats_per_warp(Dy)

    t_mask = (F_xx / Dx + F_yy / Dy) / M_mask
    t_defrag = F_xx / (Dx * M_xx) + F_yy / (Dy * M_yy)

    return t_mask / t_defrag
end


function main()
    D_min = 2
    D_max = 32
    Drange = D_min:D_max

    naive_grid  = [naive_bound(Dx, Dy)  for Dy in Drange, Dx in Drange]
    kalman_grid = [kalman_bound(Dx, Dy) for Dy in Drange, Dx in Drange]

    # Use a common colour scale so the two heatmaps are visually
    # comparable. Range chosen to match the empirical heatmap.
    clims = (0.49, 1.5)

    # p1 = heatmap(Drange, Drange, naive_grid;
    #     title  = "Naive bound (single-stage formula)",
    #     xlabel = "Dx", ylabel = "Dy",
    #     c = :viridis, clims = clims,
    #     aspect_ratio = :equal,
    #     xlims = (1.5, D_max + 0.5),
    #     ylims = (1.5, D_max + 0.5),
    #     )

    p2 = heatmap(Drange, Drange, kalman_grid;
        # title  = "Kalman-weighted bound (work-weighted FLOPs)",
        xlabel = "Dx", ylabel = "Dy",
        c = :viridis, clims = clims,
        aspect_ratio = :equal,
        xlims = (1.5, D_max + 0.5),
        ylims = (1.5, D_max + 0.5),
    )

    # plt = plot(p1, p2; layout = (1, 2), size = (1200, 500))
    plt = plot(p2, size = (400, 350))
    display(plt)
    return plt
end

path = joinpath(@__DIR__, "figs", "theoretical_bounds_kalman.png")
savefig(main(), path)
# main()
println("Wrote theoretical_bounds_kalman.png")
