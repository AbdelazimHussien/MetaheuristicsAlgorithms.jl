"""
# References: 
- Lian, Junbo, Ting Zhu, Ling Ma, Xincan Wu, Ali Asghar Heidari, Yi Chen, Huiling Chen, and Guohua Hui. 
"The educational competition optimizer." 
International Journal of Systems Science 55, no. 15 (2024): 3185-3222.
"""

function ECO(npop::Int, max_iter::Int, lb::Union{Real,AbstractVector}, ub::Union{Real,AbstractVector}, dim::Int, objfun::Function)
    function close(t::Vector{Float64}, G::Int, X::Matrix{Float64})
        m = copy(X[1, :])

        if G == 1
            for s = 1:G1Number
                school = X[s, :]
                if sum(abs.(m .- t)) > sum(abs.(school .- t))
                    m = school
                end
            end
        else
            for s = 1:G2Number
                school = X[s, :]
                if sum(abs.(m .- t)) > sum(abs.(school .- t))
                    m = school
                end
            end
        end
        return m
    end

    function initializationLogistic(pop::Int, dim::Int, ub::Union{Int,AbstractVector}, lb::Union{Int,AbstractVector})
        Boundary_no = length(ub)
        Positions = zeros(Float64, pop, dim)

        for i = 1:pop
            for j = 1:dim
                x0 = rand()
                a = 4.0
                x = a * x0 * (1 - x0)
                if Boundary_no == 1
                    Positions[i, j] = (ub[1] - lb[1]) * x + lb[1]
                    if Positions[i, j] > ub[1]
                        Positions[i, j] = ub[1]
                    end
                    if Positions[i, j] < lb[1]
                        Positions[i, j] = lb[1]
                    end
                else
                    Positions[i, j] = (ub[j] - lb[j]) * x + lb[j]
                    if Positions[i, j] > ub[j]
                        Positions[i, j] = ub[j]
                    end
                    if Positions[i, j] < lb[j]
                        Positions[i, j] = lb[j]
                    end
                end
                x0 = x
            end
        end
        return Positions
    end

    # function Levy(d::Int)
    #     beta = 1.5
    #     sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    #     u = randn(d) * sigma
    #     v = randn(d)
    #     step = u ./ abs.(v) .^ (1 / beta)
    #     return step
    # end

    H = 0.5
    G1 = 0.2
    G2 = 0.1

    G1Number = round(Int, npop * G1)
    G2Number = round(Int, npop * G2)

    if length(ub) == 1
        ub = fill(ub[1], dim)
        lb = fill(lb[1], dim)
    end

    X0 = initializationLogistic(npop, dim, ub, lb)
    X = copy(X0)

    fitness = zeros(Float64, npop)
    fitness_new = zeros(Float64, npop)
    for i = 1:npop
        fitness[i] = objfun(X[i, :])
    end

    index = sortperm(fitness)
    fitness = sort(fitness)

    GBestF = fitness[1]

    for i = 1:npop
        X[i, :] = X0[index[i], :]
    end

    curve = zeros(Float64, max_iter)
    Best_score = Inf
    Best_pos = zeros(dim)

    GBestX = X[1, :]
    GWorstX = X[end, :]
    X_new = copy(X)

    for i = 1:max_iter
        R1 = rand()
        R2 = rand()
        P = 4 * randn() * (1 - i / max_iter)
        E = (π * i) / (P * max_iter)
        w = 0.1 * log(2 - (i / max_iter))

        for j = 1:npop
            if i % 3 == 1
                if j >= 1 && j <= G1Number
                    X_new[j, :] = X[j, :] .+ w * (mean(X[j, :]) .- X[j, :]) .* levy(dim)
                else
                    X_new[j, :] = X[j, :] .+ w * (close(X[j, :], 1, X) .- X[j, :]) .* randn()
                end

            elseif i % 3 == 2
                if j >= 1 && j <= G2Number
                    X_new[j, :] = X[j, :] .+ (GBestX .- mean(X, dims=1)') * exp(i / max_iter - 1) .* levy(dim)
                else
                    if R1 < H
                        X_new[j, :] = X[j, :] .- w * close(X[j, :], 2, X) .- P * (E * w * close(X[j, :], 2, X) .- X[j, :])
                    else
                        X_new[j, :] = X[j, :] .- w * close(X[j, :], 2, X) .- P * (w * close(X[j, :], 2, X) .- X[j, :])
                    end
                end

            else
                if j >= 1 && j <= G2Number
                    X_new[j, :] = X[j, :] .+ (GBestX .- X[j, :]) .* randn() .- (GBestX .- X[j, :]) .* randn()
                else
                    if R2 < H
                        X_new[j, :] = GBestX .- P * (E * GBestX .- X[j, :])
                    else
                        X_new[j, :] = GBestX .- P * (GBestX .- X[j, :])
                    end
                end
            end

            Flag4ub = X_new[j, :] .> ub
            Flag4lb = X_new[j, :] .< lb
            X_new[j, :] = (X_new[j, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness_new[j] = objfun(X_new[j, :])
            if fitness_new[j] > fitness[j]
                fitness_new[j] = fitness[j]
                X_new[j, :] = X[j, :]
            end

            if fitness_new[j] < GBestF
                GBestF = fitness_new[j]
                GBestX = X_new[j, :]
            end
        end

        X = copy(X_new)
        fitness = copy(fitness_new)
        curve[i] = GBestF
        Best_pos = GBestX
        Best_score = curve[end]

        index = sortperm(fitness)
        fitness = fitness[index]
        for j = 1:npop
            X0[j, :] = X[index[j], :]
        end
        X = copy(X0)
    end

    return Best_score, Best_pos, curve
end