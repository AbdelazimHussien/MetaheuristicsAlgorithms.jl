"""
# References:

-  Zheng, Boli, Yi Chen, Chaofan Wang, Ali Asghar Heidari, Lei Liu, and Huiling Chen. 
"The moss growth optimization (MGO): concepts and performance." 
Journal of Computational Design and Engineering 11, no. 5 (2024): 184-221.
"""
function MossGO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return MossGO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter) 
end

function MossGO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    FEs = 0
    MaxFEs = npop * max_iter
    best_cost = Inf
    best_M = zeros(dim)
    M = initialization(npop, dim, ub, lb)
    costs = zeros(npop)

    for i = 1:npop
        costs[i] = objfun(M[i, :])
        FEs += 1
        if costs[i] < best_cost
            best_M .= M[i, :]
            best_cost = costs[i]
        end
    end

    Convergence_curve = Float64[]
    it = 1
    rec = 1

    w = 2
    rec_num = 10
    divide_num = Int(floor(dim / 4))
    d1 = 0.2

    newM = zeros(npop, dim)
    newM_cost = zeros(npop)
    rM = zeros(npop, dim, rec_num)
    rM_cos = zeros(npop, rec_num)

    while FEs < MaxFEs
        calPositions = copy(M)
        div_num = randperm(dim)

        for j = 1:max(divide_num, 1)
            th = best_M[div_num[j]]
            index = calPositions[:, div_num[j]] .> th
            if sum(index) < size(calPositions, 1) / 2
                index .= .~index
            end
            calPositions = calPositions[index, :]
        end

        D = repeat(best_M', size(calPositions, 1)) .- calPositions

        D_wind = sum(D, dims=1) ./ size(calPositions, 1)
        D_wind = D_wind'

        beta = size(calPositions, 1) / npop
        gama = 1 / sqrt(1 - beta^2)
        step = w * (rand(length(D_wind)) .- 0.5) .* (1 - FEs / MaxFEs)
        step2 = 0.1 * w * (rand(length(D_wind)) .- 0.5) .* (1 - FEs / MaxFEs) .* (1 + 1 / 2 * (1 + tanh(beta / gama)) * (1 - FEs / MaxFEs))
        step3 = 0.1 * (rand() - 0.5) * (1 - FEs / MaxFEs)
        rand_vals = rand(length(D_wind))
        result = 1 ./ (1 .+ (0.5 .- 10 .* rand_vals))
        act = actCal(result)

        if rec == 1
            rM[:, :, rec] .= M
            rM_cos[:, rec] .= costs
            rec += 1
        end

        for i = 1:npop
            newM[i, :] .= M[i, :]
            if rand() > d1
                newM[i, :] = newM[i, :] .+ step .* D_wind
            else
                newM[i, :] = newM[i, :] .+ step2 .* D_wind
            end

            if rand() < 0.8
                if rand() > 0.5
                    newM[i, div_num[1]] = best_M[div_num[1]] + step3 * D_wind[div_num[1]]
                else
                    newM[i, :] .= (1 .- act) .* newM[i, :] .+ act .* best_M
                end
            end

            Flag4ub = newM[i, :] .> ub
            Flag4lb = newM[i, :] .< lb
            newM[i, :] .= (newM[i, :] .* .~(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            newM_cost[i] = objfun(newM[i, :])
            FEs += 1

            rM[i, :, rec] .= newM[i, :]
            rM_cos[i, rec] = newM_cost[i]

            if newM_cost[i] < best_cost
                best_M .= newM[i, :]
                best_cost = newM_cost[i]
            end
        end

        rec += 1
        if rec > rec_num || FEs >= MaxFEs
            lcost, Iindex = findmin(rM_cos, dims=2)
            Iindex = map(x -> x[2], Iindex)
            for i = 1:npop
                M[i, :] = rM[i, :, Iindex[i]]
            end
            costs .= lcost
            rec = 1
        end

        push!(Convergence_curve, best_cost)
        it += 1
    end

    # return best_cost, best_M, Convergence_curve
    return OptimizationResult(
        best_M,
        best_cost,
        Convergence_curve)
end

function MossGO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return MossGO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function actCal(X)
    act = X .>= 0.5  # Apply element-wise comparison, return a boolean array
    return act
end