function polr(a, R, N)
    th = a * π * rand(N)  # Generate theta
    r = th .+ R * rand(N)  # Calculate r

    xR = r .* sin.(th)  # Element-wise sin
    yR = r .* cos.(th)  # Element-wise cos

    # Normalize
    xR ./= maximum(abs.(xR))
    yR ./= maximum(abs.(yR))

    return xR, yR
end

function swoo_p(a, R, N)
    th = a * π * exp.(rand(N))
    r = th

    xR = r .* sinh.(th)  # Element-wise sinh
    yR = r .* cosh.(th)  # Element-wise cosh

    # Normalize
    xR ./= maximum(abs.(xR))
    yR ./= maximum(abs.(yR))

    return xR, yR
end

function swoop(objfun, popPos, popCost, BestPos, BestCost, npop, lb, ub)
    Mean = mean(popPos, dims=1)
    a = 10
    R = 1.5
    s1 = 0

    # Shuffle population
    A = randperm(npop)
    popPos = popPos[A, :]
    popCost = popCost[A]

    x, y = swoo_p(a, R, npop)

    for i = 1:npop
        Step = popPos[i, :] .- 2 * Mean'
        Step1 = popPos[i, :] .- 2 * BestPos
        newsolPos = rand(length(Mean)) .* BestPos .+ x[i] * Step .+ y[i] * Step1

        newsolPos = clamp.(newsolPos, lb, ub)
        newsolPos = vec(newsolPos)
        newsolCost = objfun(newsolPos)

        if newsolCost < popCost[i]
            popPos[i, :] = newsolPos
            popCost[i] = newsolCost
            s1 += 1

            if popCost[i] < BestCost
                BestPos = popPos[i, :]
                BestCost = popCost[i]
            end
        end
    end

    return popPos, popCost, BestPos, BestCost, s1
end

function search_space(objfun, popPos, popCost, BestPos, BestCost, npop, lb, ub)
    Mean = mean(popPos, dims=1)
    a = 10
    R = 1.5
    s1 = 0

    # Shuffle population
    A = randperm(npop)
    popPos = popPos[A, :]
    popCost = popCost[A]

    x, y = polr(a, R, npop)

    for i = 1:npop-1
        Step = popPos[i, :] .- popPos[i+1, :]
        Step1 = popPos[i, :] .- Mean'
        newsolPos = popPos[i, :] .+ y[i] * vec(Step) .+ x[i] * Step1
        newsolPos = vec(newsolPos)
        newsolPos = clamp.(newsolPos, lb, ub)
        newsolCost = objfun(newsolPos)

        if newsolCost < popCost[i]
            popPos[i, :] = newsolPos
            popCost[i] = newsolCost
            s1 += 1

            if popCost[i] < BestCost
                BestPos = popPos[i, :]
                BestCost = popCost[i]
            end
        end
    end

    return popPos, popCost, BestPos, BestCost, s1
end

function select_space(objfun, popPos, popCost, npop, BestPos, BestCost, lb, ub, dim)
    Mean = mean(popPos, dims=1)
    lm = 2
    s1 = 0

    for i = 1:npop
        newsolPos = BestPos .+ lm * rand(dim) .* (vec(Mean) .- popPos[i, :])
        newsolPos = clamp.(newsolPos, lb, ub)
        newsolPos = vec(newsolPos)
        newsolCost = objfun(vec(newsolPos))

        if newsolCost < popCost[i]
            popPos[i, :] = newsolPos
            popCost[i] = newsolCost
            s1 += 1

            if popCost[i] < BestCost
                BestPos = popPos[i, :]
                BestCost = popCost[i]
            end
        end
    end

    return popPos, popCost, BestPos, BestCost, s1
end

"""
# References:
- Alsattar, Hassan A., A. A. Zaidan, and B. B. Zaidan.
  "Novel meta-heuristic bald eagle search optimisation algorithm."
  Artificial Intelligence Review 53 (2020): 2237-2264.
"""
function BES(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return BES(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function BES(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)

    # Initialize Best Solution
    BestCost = Inf
    BestPos = zeros(dim)

    # Population initialization
    # popPos = lb .+ (ub - lb) .* rand(npop, dim)
    popPos = initialization(npop, dim, ub, lb)
    popCost = [objfun(popPos[i, :]) for i = 1:npop]

    for i = 1:npop
        if popCost[i] < BestCost
            BestPos = popPos[i, :]
            BestCost = popCost[i]
        end
    end

    Convergence_curve = zeros(max_iter)

    # Main loop
    for t = 1:max_iter
        (popPos, popCost, BestPos, BestCost, s1) = select_space(objfun, popPos, popCost, npop, BestPos, BestCost, lb, ub, dim)

        (popPos, popCost, BestPos, BestCost, s2) = search_space(objfun, popPos, popCost, BestPos, BestCost, npop, lb, ub)

        (popPos, popCost, BestPos, BestCost, s3) = swoop(objfun, popPos, popCost, BestPos, BestCost, npop, lb, ub)

        # Store convergence data
        Convergence_curve[t] = BestCost
    end

    # return BestCost, BestPos, Convergence_curve
    return OptimizationResult(
        BestPos,
        BestCost,
        Convergence_curve)
end

function BES(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return BES(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end