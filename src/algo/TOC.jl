"""
# References:

- Braik, M., Al-Hiary, H., Alzoubi, H., Hammouri, A., Azmi Al-Betar, M., & Awadallah, M. A. (2025).
  Tornado optimizer with Coriolis force: a novel bio-inspired meta-heuristic algorithm for solving engineering problems.
  Artificial Intelligence Review, 58(4), 1-99.
"""
function TOC(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return TOC(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function TOC(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer, nto=4, nt=3)
    dim = length(lb)
    ccurve = zeros(max_iter)

    # Initialization
    y = initialization(npop, dim, ub, lb)
    fit = [objfun(y[i, :]) for i = 1:axes(y, 1)]
    sorted_idx = sortperm(fit)

    To = nto - nt
    nw = npop - nto

    tornadoposition = y[sorted_idx[1:To], :]
    tornadocost = fit[sorted_idx[1:To]]

    Thunderstormsposition = y[sorted_idx[2:nto], :]
    ThunderstormsCost = fit[sorted_idx[2:nto]]

    bThunderstormsCost = copy(ThunderstormsCost)
    gThunderstormsCost = zeros(nto - 1)
    ind = argmin(ThunderstormsCost)

    bThunderstormsposition = copy(Thunderstormsposition)
    gThunderstormsCost = Thunderstormsposition[ind, :]

    Windstormsposition = y[sorted_idx[nto+1:end], :]
    WindstormsCost = fit[sorted_idx[nto+1:end]]

    gWindstormsposition = zeros(nw)
    bWindstormsCost = copy(WindstormsCost)
    ind = argmin(WindstormsCost)
    bWindstormsposition = copy(Windstormsposition)
    gWindstormsposition = Windstormsposition[ind, :]

    vel_storm = 0.1 .* Windstormsposition

    nwindstorms = sort(randperm(nw)[1:nto])
    nWT = diff([0; nwindstorms])
    push!(nWT, nw - sum(nWT))

    nWT1 = nWT[1]
    nWH = nWT[2:end]

    b_r = 100000
    fdelta = [-1, 1]

    chi = 4.10
    eta = 2 / abs(2 - chi - sqrt(chi^2 - 4 * chi))

    t = 1
    println("================  Tornado Optimizer with Coriolis force (TOC) ================ ")

    while t <= max_iter
        nu = (0.1 * exp(-0.1 * (t / max_iter)^0.1))^16
        mu = 0.5 + rand() / 2
        ay = (max_iter - (t^2 / max_iter)) / max_iter

        Rl = 2 / (1 + exp((-t + max_iter / 2) / 2))
        Rr = -2 / (1 + exp((-t + max_iter / 2) / 2))

        # TODO: Continue porting core update loops (windstorms to tornado, exploitation, etc.)

        println("Iteration: $t   mintornadocost= $(minimum(tornadocost))")
        ccurve[t] = minimum(tornadocost)
        t += 1
    end

    # return tornadocost, tornadoposition, ccurve
    return OptimizationResult(
        tornadoposition,
        tornadocost,
        ccurve)
end


function TOC(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return TOC(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end