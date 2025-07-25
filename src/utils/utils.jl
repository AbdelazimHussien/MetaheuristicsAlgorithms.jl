
addpath(directory) =
    map(file -> endswith(file, ".jl") && include(joinpath(directory, file)), readdir(directory))

function levy(d::Integer)
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    u = randn(d) * sigma
    v = randn(d)
    step = @. u / abs(v)^(1 / beta)
    return step
end

function levy(n::Integer, m::Integer, beta::AbstractFloat)
    num = gamma(1 + beta) * sin(pi * beta / 2)

    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)

    sigma_u = (num / den)^(1 / beta)

    u = rand(Normal(0, sigma_u), n, m)

    v = rand(Normal(0, 1), n, m)

    z = @. u / abs(v)^(1 / beta)

    return z
end

function levy(beta_base::AbstractFloat, dim::Integer, t::Integer, max_iter::Integer)
    beta = beta_base + 0.5 * sin(pi / 2 * t / max_iter)
    beta = clamp(beta, 0.1, 2.0)
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    u = randn(dim) * sigma^2
    v = randn(dim)
    w = abs.(u ./ abs.(v) .^ (1 / beta))
    return w ./ (maximum(w) + eps())
end

function clamp_matrix!(x::AbstractMatrix, lb::AbstractVector, ub::AbstractVector)
    @inbounds for j in axes(x, 2)
        @simd for i in axes(x, 1)
            x[i, j] = ifelse(x[i, j] < lb[i], lb[i], x[i, j])
            x[i, j] = ifelse(x[i, j] > ub[i], ub[i], x[i, j])
        end
    end
    return x
end
