"""
# References:

- Dhiman, Gaurav, and Amandeep Kaur.
  "STOA: a bio-inspired based optimization algorithm for industrial engineering problems."
  Engineering Applications of Artificial Intelligence 82 (2019): 148-174.
"""
function STOA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return STOA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end
function STOA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    position = zeros(Float64, dim)
    score = Inf

    positions = initialization(npop, dim, ub, lb)
    convergence = zeros(Float64, max_iter)

    l = 0

    while l < max_iter
        for i in axes(positions, 1)
            positions .= max.(lb, min.(positions, ub))

            fitness = objfun(positions[i, :])

            if fitness < score
                score = fitness
                position .= positions[i, :]
            end
        end

        Sa = 2 - l * (2 / max_iter)

        for i in axes(positions, 1)
            for j in axes(positions, 2)
                r1 = 0.5 * rand()
                r2 = 0.5 * rand()
                X1 = 2 * Sa * r1 - Sa
                b = 1
                ll = (Sa - 1) * rand() + 1
                D_alphs = Sa * positions[i, j] + X1 * (position[j] - positions[i, j])
                res = D_alphs * exp(b * ll) * cos(ll * 2 * π) + position[j]
                positions[i, j] = res
            end
        end
        l += 1
        convergence[l] = score
    end

    # return score, position, convergence
    return OptimizationResult(
        position,
        score,
        convergence)
end

function STOA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return STOA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end