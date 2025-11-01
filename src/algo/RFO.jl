"""
# References:

- Braik, M., & Al-Hiary, H. (2025).
  Rüppell’s fox optimizer: A novel meta-heuristic approach for solving global optimization problems.
  Cluster Computing, 28(5), 1-77.
"""
function RFO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return RFO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function RFO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # Preallocate
    ccurve = zeros(max_iter)
    vec_flag = [1, -1]
    beta = 1e-10
    L = 100.0

    # Initialize population
    fposition = initialization(npop, dim, ub, lb)
    x = copy(fposition)

    # Evaluate initial fitness
    fit = zeros(npop)
    for i = 1:npop
        fit[i] = objfun(view(fposition, i, :))
    end
    fitness = copy(fit)

    # Initialize bests
    fmin0, idx = findmin(fit)
    pbest = copy(fposition)
    fbest = copy(pbest[idx, :])

    # Main loop
    for ite = 1:max_iter
        # Adaptive parameters
        h = 1.0 / (1 + exp((ite - max_iter / 2) / L))
        s = 1.0 / (1 + exp((max_iter / 2 - ite) / L))
        tmp = 2.0 / (1 + exp((max_iter / 2 - ite) / 100))
        smell = 0.1 / abs(acos(clamp(tmp, -1.0, 1.0)))

        # Daylight vs Night
        if rand() >= 0.5
            # Daylight
            ds = rand(1:npop, npop)
            for i = 1:npop
                if s >= h
                    if rand() >= 0.25
                        Xr = view(pbest, ds[i], :)
                        ri = floor(Int, 4 * rand() + 1)
                        r = rand()
                        fposition[i, :] .= x[i, :] .+ ri * r * (Xr .- x[i, :]) .+ ri * r * (fbest .- x[i, :])
                        θ = rand() * 260 * π / 180
                        fposition[i, :] .= frotate(fposition[i, :], x[i, :], θ, dim)
                    else
                        Flag = vec_flag[rand(1:2)]
                        Xr = view(pbest, ds[i], :)
                        fposition[i, :] .= Xr .+ beta * randn(dim) * Flag
                    end
                else
                    if rand() >= 0.75
                        Xr = view(pbest, ds[i], :)
                        ri = floor(Int, 4 * rand() + 1)
                        r = rand()
                        fposition[i, :] .= x[i, :] .+ ri * r * (Xr .- x[i, :]) .+ ri * r * (fbest .- x[i, :])
                        θ = rand() * 150 * π / 180
                        fposition[i, :] .= frotate(fposition[i, :], x[i, :], θ, dim)
                    else
                        Flag = vec_flag[rand(1:2)]
                        fposition[i, :] .= view(pbest, ds[i], :) .+ beta * randn(dim) * Flag
                    end
                end
            end
        else
            # Night
            dr = rand(1:npop, npop)
            for i = 1:npop
                if h >= s
                    if rand() >= 0.25
                        Xr = view(pbest, dr[i], :)
                        ri = floor(Int, 4 * rand() + 1)
                        r = rand()
                        fposition[i, :] .= x[i, :] .+ ri * r * (Xr .- x[i, :]) .+ ri * r * (fbest .- x[i, :])
                        θ = rand() * 150 * π / 180
                        fposition[i, :] .= frotate(fposition[i, :], x[i, :], θ, dim)
                    else
                        Flag = vec_flag[rand(1:2)]
                        fposition[i, :] .= view(pbest, dr[i], :) .+ beta * randn(dim) * Flag
                    end
                else
                    if rand() >= 0.75
                        Xr = view(pbest, dr[i], :)
                        ri = floor(Int, 4 * rand() + 1)
                        r = rand()
                        fposition[i, :] .= x[i, :] .+ ri * r * (Xr .- x[i, :]) .+ ri * r * (fbest .- x[i, :])
                        θ = rand() * 260 * π / 180
                        fposition[i, :] .= frotate(fposition[i, :], x[i, :], θ, dim)
                    else
                        Flag = vec_flag[rand(1:2)]
                        fposition[i, :] .= view(pbest, dr[i], :) .+ beta * randn(dim) * Flag
                    end
                end
            end
        end

        # Smell behavior
        ss = rand(1:npop, npop)
        for i = 1:npop
            if rand() >= smell
                Xr1 = view(pbest, ss[i], :)
                xr = 2 + rand() * (4 - 2)
                eps = abs(4 * rand() - (rand() + rand())) / xr
                fposition[i, :] .= x[i, :] .+ eps * (Xr1 .- x[i, :]) .+ eps * (fbest .- x[i, :])
            else
                Flag = vec_flag[rand(1:2)]
                fposition[i, :] .= view(pbest, ss[i], :) .+ beta * randn(dim) * Flag
            end
        end

        # Animal behavior
        ab = rand(1:npop, npop)
        for i = 1:npop
            if rand() >= h
                fposition[i, :] .= view(pbest, ab[i], :) .+ beta * randn(dim)
            else
                Xrp = view(pbest, ab[i], :)
                Dista = 2 * (fbest .- fposition[i, :]) * rand()
                Distb = 3 * (Xrp .- view(pbest, i, :)) * rand()
                Flag = vec_flag[rand(1:2)]
                if i == 1
                    fposition[i, :] .= fposition[i, :] .+ Dista .+ Distb * Flag
                else
                    tmpv = fposition[i, :] .+ Dista .+ Distb * Flag
                    fposition[i, :] .= (tmpv .+ fposition[i-1, :]) / 2
                end
            end
        end

        # Worst-case & boundary
        for i = 1:npop
            if vec_flag[rand(1:2)] == 1
                _, gw = findmax(fitness)
                fposition[gw, :] .+= beta * randn(dim)
            end
            fposition[i, :] .= clamp.(fposition[i, :], lb, ub)
        end

        # Update fitness & bests
        for i = 1:npop
            x[i, :] .= fposition[i, :]
            fit[i] = objfun(view(fposition, i, :))
            if fit[i] < fitness[i]
                pbest[i, :] .= fposition[i, :]
                fitness[i] = fit[i]
            end
            if fitness[i] < fmin0
                fmin0 = fitness[i]
                fbest .= pbest[i, :]
            end
        end

        ccurve[ite] = fmin0
    end

    # return fmin0, fbest, ccurve
    return OptimizationResult(
        fbest,
        fmin0,
        ccurve)
end

function RFO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return RFO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function frotate(x::Vector{T}, y::Vector{T}, θ::T, dim::Int) where {T<:Real}
    # center index
    cent = ceil(Int, dim / 2)

    # build 2×N matrix of points (each column is a point)
    v = [x'; y']  # 2×N

    # center of rotation
    x0, y0 = x[cent], y[cent]
    center = repeat([x0; y0], 1, length(x))  # 2×N

    # 2×2 rotation matrix
    R = [cos(θ) -sin(θ);
        sin(θ) cos(θ)]

    # apply rotation: shift to origin, rotate, shift back
    vo = R * (v .- center) .+ center

    # return the first row (rotated x-coordinates)
    return vec(vo[1, :])
end