"""
Yadav, Anupam. 
"AEFA: Artificial electric field algorithm for global optimization." 
Swarm and Evolutionary Computation 48 (2019): 93-108.
"""

using Statistics
using LinearAlgebra

get_population!(x, lb, ub) = x .= (ub .- lb) .* rand!(x) .+ lb
get_population(dims, lb, ub) = (ub .- lb) .* rand(Float64, dims) .+ lb

mutable struct AEFA
    npop::Int
    i::Int
    maxiters::Int
    lb::Vector{Float64}
    ub::Vector{Float64}
    dim::Int
    x::Matrix{Float64}
    v::Matrix{Float64}
    update::Bool
    #-------------------
    Rnorm::Int
    FCheck::Int
    Rpower::Int
    # tag::Int
    alfa::Int
    K0::Int
    gbest::Vector{Float64}
    gbest_val::Vector{Float64}
    mean_val::Vector{Float64}
    function AEFA(npop::Integer, maxiters::Integer, lb::AbstractVector, ub::AbstractVector)
        Rnorm = 2
        FCheck = 1
        Rpower = 1
        alfa = 30
        K0 = 500
        dim = length(lb)
        gbest = zeros(dim)

        gbest_val = Float64[]
        mean_val = Float64[]

        x = get_population((dim, npop), lb, ub)
        v = zeros(dim, npop)
        i = 1
        update = true
        new(npop, i, maxiters, lb, ub, dim, x, v, update, Rnorm, FCheck, Rpower, alfa, K0,
            gbest, gbest_val, mean_val)
    end
end

function update_state!(self::AEFA, fitness::AbstractVector)
    (; npop, i, maxiters, lb, ub, dim, x, v, Rnorm, FCheck, Rpower, alfa, K0, gbest,
        gbest_val, mean_val) = self

    best, best_X = findmin(fitness)
    worst = maximum(fitness)
    push!(mean_val, mean(fitness))

    if i == 1 || best < gbest_val[end]
        push!(gbest_val, best)
        gbest .= @view x[:, best_X]
    end

    # Stopping criteria
    if i >= maxiters
        self.update = false
        return self
    end

    Q = if best == worst
        ones(npop)
    else
        exp.((fitness .- worst) ./ (best - worst))
    end

    Q ./= sum(Q)

    fper = 3
    cbest = FCheck == 1 ? round(Int, npop * (fper + (1 - i / maxiters) * (100 - fper)) / 100) :
            npop
    s = sortperm(Q, rev=true)
    # Qs = Q[s]

    E = zeros(dim, npop)
    for i = 1:npop
        for ii = 1:cbest
            j = s[ii]
            if j != i
                R = norm(x[:, i] - x[:, j], Rnorm)
                for k = 1:dim
                    E[k, i] += rand() * Q[j] * ((x[k, j] - x[k, i]) / (R^Rpower + eps()))
                end
            end
        end
    end

    a = E * K0 * exp(-alfa * i / maxiters)

    v .*= rand(dim, npop)
    v .+= a
    x .+= v
    x .= clamp.(x, lb, ub)

    # swarm = zeros(npop, 1, 2)
    # swarm[:, 1, 1] = x[:, 1]
    # swarm[:, 1, 2] = x[:, 2]

    self.i = i + 1
    return self
end