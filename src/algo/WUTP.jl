"""
# References:

- Braik, M., & Al-Hiary, H. (2025).
  A novel meta-heuristic optimization algorithm inspired by water uptake and transport in plants.
  Neural Computing and Applications, 1-82.
"""
function WUTP(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return WUTP(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function WUTP(objfun, lb::Vector{Float64}, ub::Vector{Float64}, no_particles::Integer, max_iter::Integer)
    dim = length(lb)
    # Preallocate
    ccurve = zeros(max_iter)
    # Initialize population
    y = initialization(no_particles, dim, ub, lb)
    # Initial velocity
    J = 0.1 .* copy(y)
    # Evaluate fitness

    # cost = [objfun(view(y, i, :)) for i in 1:no_particles]
    cost = zeros(no_particles)
    for i = 1:no_particles
        cost[i] = objfun(y[i, :])
    end
    fitness = copy(cost)
    # Initial best
    fmin0, idx0 = findmin(cost)
    yo = copy(y)
    pbest = copy(y)
    gbest = copy(y[idx0, :])
    # Parameters
    p = 0.5
    rho = 1000.0
    eta = 0.0018
    g = 9.81
    Lp = 1e-9
    D = 1e-9
    K = 1e-9
    a = 1.0

    for t = 1:max_iter
        λ = 1.0 - rand() / 2
        ν = 1e-7 / (1 + exp((t - max_iter / 2) / 100))
        rr = 0.1
        # Water in motion
        for i = 1:no_particles
            k = rand(1:no_particles)
            pb = view(pbest, k, :)
            c1, c2 = rand(), rand()
            J[i, :] .= λ .* J[i, :] .+ c1 * Lp * (y[i, :] .- pb) .+ c2 * rho * g * (y[i, :] .- yo[i, :])
        end
        # Horizontal flow
        for i = 1:no_particles, j = 1:dim
            r3, r4, r5 = rand(), rand(), rand()
            if r3 < p
                if r4 > 0.1
                    y[i, j] -= J[i, j] ./ K
                else
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            else
                if r5 > 0.1
                    y[i, j] -= J[i, j] / (2 * D * pi * a^2)
                else
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            end
        end
        # Vertical flow
        for i = 1:no_particles, j = 1:dim
            if rand() > p
                if rand() > 0.1
                    y[i, j] -= J[i, j] / K + K * rho * g * (y[i, j] - yo[i, j]) * rand()
                else
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            else
                if rand() > 0.1
                    y[i, j] -= J[i, j] / (2 * D * pi * a^2) + D * rho * g * (y[i, j] - yo[i, j]) * rand()
                else
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            end
        end
        # Uptake by roots
        for j = 1:dim, i = 1:no_particles
            k = rand(1:no_particles)
            pb = pbest[k, :]
            if rand() > p
                y[i, j] -= J[i, j] / K + K * rho * g * (y[i, j] - yo[i, j]) * rand()
            else
                y[i, j] -= J[i, j] / (K * pi * a^2) + rand() * (pb[j] - yo[i, j])
            end
        end
        # Root to xylem
        for j = 1:dim, i = 1:no_particles
            k = rand(1:no_particles)
            pb = pbest[k, :]
            if rand() < 0.1
                y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
            else
                chi = 0.5
                y[i, j] -= J[i, j] / Lp + chi * (y[i, j] - yo[i, j]) * rand() + (pb[j] - y[i, j]) * rand()
            end
        end
        # Xylem to leaves
        for i = 1:no_particles
            if rand() < rr
                for j = 1:dim
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            else
                y[i, :] .+= J[i, :] .* (8 * eta / (pi * a^4))
            end
        end
        # Bound check
        for i = 1:no_particles, j = 1:dim
            y[i, j] = clamp(y[i, j], lb[j], ub[j])
        end
        # Update pbest & gbest
        for i = 1:no_particles
            cost[i] = objfun(view(y, i, :))
            if all(lb .<= y[i, :]) && all(y[i, :] .<= ub)
                yo[i, :] = y[i, :]
                if cost[i] < fitness[i]
                    pbest[i, :] = y[i, :]
                    fitness[i] = cost[i]
                end
            end
        end
        fmin, idx = findmin(fitness)
        if fmin < fmin0
            fmin0 = fmin
            gbest = pbest[idx, :]
        end
        ccurve[t] = fmin0
    end
    # return fmin0, gbest, ccurve
    return OptimizationResult(
        gbest,
        fmin0,
        ccurve)
end

function WUTP(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return WUTP(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end