"""
# References: 

- Braik, Malik, Alaa Sheta, and Heba Al-Hiary. "A novel meta-heuristic search algorithm for solving optimization problems: capuchin search algorithm." Neural computing and applications 33, no. 7 (2021): 2515-2547.

"""
function CapSA(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return CapSA(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function CapSA(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    cg_curve = zeros(max_iter)

    CapPos = initialization(npop, dim, ub, lb)

    v = 0.1 .* CapPos
    v0 = zeros(npop, dim)
    CapFit = zeros(npop)

    for i = 1:npop
        CapFit[i] = objfun(CapPos[i, :])
    end

    Fit = copy(CapFit)

    fitCapSA, index = findmin(CapFit)
    CapBestPos = copy(CapPos)
    gFoodPos = CapPos[index, :]

    bf = 0.70
    cr = 11.0
    g = 9.81

    a1 = 1.250
    a2 = 1.5

    beta = [2, 11, 2]
    wmax = 0.8
    wmin = 0.1

    for t = 1:max_iter
        tau = beta[1] * exp(-beta[2] * t / max_iter)^beta[3]
        w = wmax - (wmax - wmin) * (t / max_iter)
        fol = ceil.(Int, (npop - 1) .* rand(npop))

        for i = 1:npop
            for j = 1:dim
                v[i, j] = w * v[i, j] + a1 * (CapBestPos[i, j] - CapPos[i, j]) * rand() + a2 * (gFoodPos[j] - CapPos[i, j]) * rand()
            end
        end

        for i = 1:npop
            r = rand()
            if i < npop / 2
                if rand() >= 0.1
                    if r <= 0.15
                        CapPos[i, :] = gFoodPos + bf * ((v[i, :] .^ 2) * sin(2 * rand() * 1.5)) / g
                    elseif r > 0.15 && r <= 0.30
                        CapPos[i, :] = gFoodPos + cr * bf * ((v[i, :] .^ 2) * sin(2 * rand() * 1.5)) / g
                    elseif r > 0.30 && r <= 0.9
                        CapPos[i, :] = CapPos[i, :] + v[i, :]
                    elseif r > 0.9 && r <= 0.95
                        CapPos[i, :] = gFoodPos .+ bf * sin(rand() * 1.5)
                    elseif r > 0.95
                        CapPos[i, :] = gFoodPos .+ bf * (v[i, :] - v0[i, :])
                    end
                else
                    CapPos[i, :] = tau * (lb .+ rand(dim) .* (ub .- lb))
                end
            elseif i >= npop / 2 && i <= npop
                eps = ((rand() + 2 * rand()) - (3 * rand())) / (1 + rand())
                Pos = gFoodPos .+ 2 * (CapBestPos[fol[i], :] - CapPos[i, :]) * eps .+ 2 * (CapPos[i, :] - CapBestPos[i, :]) * eps
                CapPos[i, :] = (Pos .+ CapPos[i-1, :]) ./ 2
            end
        end

        v0 = copy(v)

        for i = 1:npop
            u = ub .- CapPos[i, :] .< 0
            l = lb .- CapPos[i, :] .> 0
            CapPos[i, :] = lb .* l .+ ub .* u .+ CapPos[i, :] .* .~xor.(u, l)
            CapFit[i] = objfun(CapPos[i, :])

            if CapFit[i] < Fit[i]
                CapBestPos[i, :] = CapPos[i, :]
                Fit[i] = CapFit[i]
            end
        end

        fmin, index = findmin(Fit)

        if fmin < fitCapSA
            gFoodPos = CapBestPos[index, :]
            fitCapSA = fmin
        end

        cg_curve[t] = fitCapSA
    end

    # return fitCapSA, gFoodPos, cg_curve
    return OptimizationResult(
        gFoodPos,
        fitCapSA,
        cg_curve)
end

function CapSA(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return CapSA(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end