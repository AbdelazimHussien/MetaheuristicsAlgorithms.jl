"""
# References:

- Yadav, Anupam.
  "AEFA: Artificial electric field algorithm for global optimization."
  Swarm and Evolutionary Computation 48 (2019): 93-108.
"""
function AEFA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return AEFA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function AEFA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    FCheck, Rpower, tag = 1, 1, 1

    Rnorm = 2

    X = initialization(npop, dim, ub, lb)
    V = zeros(npop, dim)
    BestValues = Float64[]
    MeanValues = Float64[]

    Fbest = nothing
    Lbest = zeros(dim)

    for iteration = 1:max_iter
        # Evaluate fitness
        fitness = [objfun(X[i, :]) for i = 1:npop]

        if tag == 1
            best, best_X = findmin(fitness)
        else
            best, best_X = findmax(fitness)
        end

        if iteration == 1 || (tag == 1 && best < Fbest) || (tag != 1 && best > Fbest)
            Fbest = best
            Lbest = copy(X[best_X, :])
        end

        push!(BestValues, Fbest)
        push!(MeanValues, mean(fitness))

        # Charge computation
        Fmax = maximum(fitness)
        Fmin = minimum(fitness)
        Fmean = mean(fitness)

        if Fmax == Fmin
            Q = ones(N)
        else
            best_f = tag == 1 ? Fmin : Fmax
            worst_f = tag == 1 ? Fmax : Fmin
            Q = exp.((fitness .- worst_f) ./ (best_f - worst_f))
        end

        Q = Q ./ sum(Q)

        # Calculate total electric force
        fper = 3.0
        cbest = FCheck == 1 ? round(Int, npop * (fper + (1 - iteration / max_iter) * (100 - fper)) / 100) : npop
        s = sortperm(Q, rev=true)

        E = zeros(npop, dim)
        for i = 1:npop
            for ii = 1:cbest
                j = s[ii]
                if j != i
                    R = norm(X[i, :] - X[j, :], Rnorm)
                    for k = 1:dim
                        E[i, k] += rand() * Q[j] * (X[j, k] - X[i, k]) / (R^Rpower + eps())
                    end
                end
            end
        end

        # Coulomb constant
        alfa = 30.0
        K0 = 500.0
        K = K0 * exp(-alfa * iteration / max_iter)

        a = E * K
        V = rand(npop, dim) .* V .+ a
        X = X .+ V

        X = clamp.(X, lb, ub)

        # Plotting
        # if D >= 2
        #     swarm = hcat(X[:, 1], X[:, 2])
        #     scatter(swarm[:, 1], swarm[:, 2], label="Particles", color=:blue, marker=:x, legend=false)
        #     scatter!([X[best_X, 1]], [X[best_X, 2]], color=:red, marker=:star5, label="Best")
        #     title!("Iteration: $iteration")
        #     xlims!(lb, ub)
        #     ylims!(lb, ub)
        #     sleep(0.2)
        # end
    end

    println("Fbest ", Fbest)
    println("Lbest ", Lbest)
    return Fbest, Lbest, BestValues, MeanValues
    # return OptimizationResult(
    #     Lbest,
    #     Fbest,
    #     BestValues)
end

function AEFA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)#::OptimizationResult
    return AEFA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end