"""
# References:

- Xue J, Shen B. Dung beetle optimizer: A new meta-heuristic algorithm for global optimization.
  The Journal of Supercomputing. 2023 May; 79(7):7305-36.
"""
function DBO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return DBO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function DBO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    P_percent = 0.2
    pNum = round(Int, npop * P_percent)

    # x = [lb .+ (ub .- lb) .* rand(dim) for _ = 1:npop]
    x = initialization(npop, dim, ub, lb)
    fit = [objfun(x[i]) for i = 1:npop]
    pFit = copy(fit)
    pX = copy(x)
    XX = copy(pX)

    fMin, bestI = findmin(fit)
    bestX = copy(x[bestI])
    Convergence_curve = zeros(max_iter)

    for t = 1:max_iter
        fmax, B = findmax(fit)
        worse = x[B]
        r2 = rand()

        for i = 1:pNum
            if r2 < 0.9
                r1 = rand()
                a = rand() > 0.1 ? 1 : -1
                x[i] = pX[i] .+ 0.3 .* abs.(pX[i] .- worse) .+ a * 0.1 .* XX[i]
            else
                theta = rand([0, 90, 180]) * π / 180
                x[i] = pX[i] .+ tan(theta) .* abs.(pX[i] .- XX[i])
            end
            x[i] = clamp.(x[i], lb, ub)
            fit[i] = objfun(x[i])
        end

        fMMin, bestII = findmin(fit)
        bestXX = copy(x[bestII])
        R = 1 - t / max_iter

        Xnew1 = clamp.(bestXX .* (1 - R), lb, ub)
        Xnew2 = clamp.(bestXX .* (1 + R), lb, ub)
        Xnew11 = clamp.(bestX .* (1 - R), lb, ub)
        Xnew22 = clamp.(bestX .* (1 + R), lb, ub)

        for i = pNum+1:12
            x[i] = bestXX .+ rand(dim) .* (pX[i] .- Xnew1) .+ rand(dim) .* (pX[i] .- Xnew2)
            x[i] = clamp.(x[i], Xnew1, Xnew2)
            fit[i] = objfun(x[i])
        end

        for i = 13:19
            x[i] = pX[i] .+ randn() .* (pX[i] .- Xnew11) .+ rand(dim) .* (pX[i] .- Xnew22)
            x[i] = clamp.(x[i], lb, ub)
            fit[i] = objfun(x[i])
        end

        for j = 20:npop
            x[j] = bestX .+ randn(dim) .* ((abs.(pX[j] .- bestXX) .+ abs.(pX[j] .- bestX)) ./ 2)
            x[j] = clamp.(x[j], lb, ub)
            fit[j] = objfun(x[j])
        end

        XX = copy(pX)
        for i = 1:npop
            if fit[i] < pFit[i]
                pFit[i] = fit[i]
                pX[i] = copy(x[i])
            end
            if pFit[i] < fMin
                fMin = pFit[i]
                bestX = copy(pX[i])
            end
        end
        Convergence_curve[t] = fMin
    end

    # return fMin, bestX, Convergence_curve
    return OptimizationResult(
        bestX,
        fMin,
        Convergence_curve)
end

function DBO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return DBO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end