"""
Yuan, Chong, Dong Zhao, Ali Asghar Heidari, Lei Liu, Yi Chen, and Huiling Chen. 
"Polar lights optimizer: Algorithm and applications in image segmentation and feature selection." 
Neurocomputing 607 (2024): 128427.
"""


function PLO(N::Int, MaxIt::Int, lb, ub, dim::Int, fobj::Function)
    FEs = 0
    it = 1
    MaxFEs = N * MaxIt
    fitness = fill(Inf, N)
    fitness_new = fill(Inf, N)

    X = initialization(N, dim, ub, lb)
    V = ones(N, dim)
    X_new = zeros(N, dim)

    for i in 1:N
        fitness[i] = fobj(X[i, :])
        FEs += 1
    end

    SortOrder = sortperm(fitness)
    fitness = fitness[SortOrder]
    
    X = X[SortOrder, :]
    Bestpos = X[1, :]
    Bestscore = fitness[1]

    Convergence_curve = Float64[]
    push!(Convergence_curve, Bestscore)

    while FEs <= MaxFEs
        X_sum = sum(X, dims=1)
        X_mean = vec(X_sum / N)
        w1 = tanh((FEs / MaxFEs)^4)
        w2 = exp(-(2 * FEs / MaxFEs)^3)

        for i in 1:N
            a = rand() / 2 + 1
            V[i, :] .= 1 * exp((1 - a) / 100 * FEs)
            LS = V[i, :]

            GS = vec(Levy(dim)) .* (X_mean .- X[i, :] + (lb .+ rand(dim) .* (ub .- lb)) / 2)
            X_new[i, :] = X[i, :] .+ (w1 * LS .+ w2 * GS) .* rand(dim)
        end

        E = sqrt(FEs / MaxFEs)
        A = randperm(N)

        for i in 1:N
            for j in 1:dim
                if (rand() < 0.05) && (rand() < E)
                    X_new[i, j] = X[i, j] + sin(rand() * π) * (X[i, j] - X[A[i], j])
                end
            end

            Flag4ub = X_new[i, :] .> ub
            Flag4lb = X_new[i, :] .< lb
            X_new[i, :] = X_new[i, :] .* .~(Flag4ub .+ Flag4lb) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness_new[i] = fobj(X_new[i, :])
            FEs += 1

            if fitness_new[i] < fitness[i]
                X[i, :] = X_new[i, :]
                fitness[i] = fitness_new[i]
            end
        end

        SortOrder = sortperm(fitness)
        fitness = fitness[SortOrder]

        X = X[SortOrder, :]
        if fitness[1] < Bestscore
            Bestpos = X[1, :]
            Bestscore = fitness[1]
        end

        it += 1
        println("$it ---> $Bestscore")
        push!(Convergence_curve, Bestscore)
    end

    return Bestscore, Bestpos, Convergence_curve
end