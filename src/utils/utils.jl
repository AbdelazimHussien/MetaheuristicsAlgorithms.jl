
addpath(directory) =
    map(file -> endswith(file, ".jl") && include(joinpath(directory, file)), readdir(directory))

get_population!(x, lb, ub) = x .= (ub .- lb) .* rand!(x) .+ lb
get_population(dims, lb, ub) = (ub .- lb) .* rand(Float64, dims) .+ lb

function Levy(d::Int)
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    u = randn(d) * sigma
    v = randn(d)
    step = u ./ abs.(v) .^ (1 / beta)
    return step
end

function Levy(n::Int, m::Int, beta::Float64)
    num = gamma(1 + beta) * sin(pi * beta / 2)

    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)

    sigma_u = (num / den)^(1 / beta)

    u = rand(Normal(0, sigma_u), n, m)

    v = rand(Normal(0, 1), n, m)

    z = u ./ abs.(v) .^ (1 / beta)

    return z
end