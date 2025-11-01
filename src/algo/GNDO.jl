"""
# References:
- Zhang, Yiying, Zhigang Jin, and Seyedali Mirjalili.
  "Generalized normal distribution optimization and its applications in parameter extraction of photovoltaic models."
  Energy Conversion and Management 224 (2020): 113301.
"""
function GNDO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return GNDO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function GNDO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # x = lb .+ (ub .- lb) .* rand(npop, dim)
    x = initialization(npop, dim, ub, lb)

    bestFitness = Inf
    bestSol = zeros(dim)
    cgcurve = zeros(max_iter)

    fitness = zeros(npop)

    for it = 1:max_iter
        for i = 1:npop
            fitness[i] = objfun(x[i, :])
        end

        for i = 1:npop
            if fitness[i] < bestFitness
                bestSol = x[i, :]
                bestFitness = fitness[i]
            end
        end
        cgcurve[it] = bestFitness

        mo = mean(x, dims=1)

        for i = 1:npop
            a, b, c = randperm(npop)[1:3]
            while a == i || b == i || c == i || a == b || a == c || b == c
                a, b, c = randperm(npop)[1:3]
            end

            v1 = (fitness[a] < fitness[i]) ? (x[a] - x[i]) : (x[i] - x[a])
            v2 = (fitness[b] < fitness[c]) ? (x[b] - x[c]) : (x[c] - x[b])

            if rand() <= rand()
                u = (1 / 3) * (x[i, :] + bestSol + mo')

                deta = sqrt.((1 / 3) .* ((x[i, :] - u) .^ 2 + (bestSol - u) .^ 2 + vec(mo' - u)) .^ 2)

                vc1 = rand(dim)
                vc2 = rand(dim)

                Z1 = sqrt.(-log.(vc2)) .* cos.(2pi .* vc1)
                Z2 = sqrt.(-log.(vc2)) .* cos.(2pi .* vc1 .+ pi)

                a_rand = rand()
                b_rand = rand()

                if a_rand <= b_rand
                    eta = u + deta .* Z1
                else
                    eta = u + deta .* Z2
                end

                newsol = eta
            else
                beta = rand()
                v = x[i] .+ beta .* abs.(randn(dim)) .* v1 .+ (1 - beta) .* abs.(randn(dim)) .* v2
                newsol = v
            end

            newsol = clamp.(newsol, lb, ub)
            newfitness = objfun(vec(newsol))

            if newfitness < fitness[i]
                x[i, :] = newsol
                fitness[i] = newfitness

                if fitness[i] < bestFitness
                    bestSol = x[i, :]
                    bestFitness = fitness[i]
                end
            end
        end

        cgcurve[it] = bestFitness
    end

    # return bestFitness, bestSol, cgcurve
    return OptimizationResult(
        bestSol,
        bestFitness,
        cgcurve)
end

function GNDO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return GNDO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end