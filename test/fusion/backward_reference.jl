# Independent dense Gaussian reference: stack all future observations and their
# noise loadings conditional on the state at the first observation.
function backward_joint_reference(models)
    H, c, UR, y, A, b, UQ = first(models)
    n = size(H, 2)
    width = sum(size(x[3], 1) + size(x[7], 1) for x in models)
    F = Matrix{Float64}(I, n, n)
    offset = zeros(n)
    noise = zeros(n, width)
    G = zeros(0, n)
    h = Float64[]
    E = zeros(0, width)
    obs = Float64[]
    col = 0
    for (t, (H, c, UR, y, A, b, UQ)) in enumerate(models)
        if t > 1
            F = A * F
            offset = A * offset + b
            noise = A * noise
            q = size(UQ, 1)
            noise[:, (col + 1):(col + q)] = UQ'
            col += q
        end
        m = length(y)
        observation_noise = H * noise
        observation_noise[:, (col + 1):(col + m)] = UR'
        col += m
        G = vcat(G, H * F)
        h = vcat(h, H * offset + c)
        E = vcat(E, observation_noise)
        obs = vcat(obs, y)
    end
    return G, h, E * E', obs
end
function dense_logpdf(residual, covariance)
    C = cholesky(Symmetric(covariance))
    return -(length(residual) * log(2π) + logdet(C) + dot(residual, C \ residual)) / 2
end
function dense_overlap(μ, P, B, r, c=0)
    V = I + B * P * B'
    e = r - B * μ
    return c - (logdet(Symmetric(V)) + dot(e, V \ e)) / 2
end
