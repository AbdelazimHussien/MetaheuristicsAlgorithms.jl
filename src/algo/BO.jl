"""
# References:

- Das, Amit Kumar, and Dilip Kumar Pratihar. "Bonobo optimizer (BO): an intelligent heuristic with self-adjusting parameters over continuous spaces and its applications to engineering problems." Applied Intelligence 52, no. 3 (2022): 2942-2974.

"""
function BO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return BO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function BO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    p_xgm_initial = 0.001
    scab = 1.25
    scsb = 1.3
    rcpp = 0.0035
    tsgs_factor_max = 0.05
    if length(lb) == 1
        lb = lb * ones(dim)
    end

    if length(ub) == 1
        ub = ub * ones(dim)
    end

    # Initialization
    cost = zeros(npop)
    bonobo = zeros(npop, dim)
    convergence = zeros(max_iter)
    for i = 1:npop
        bonobo[i, :] = [rand(Uniform(lb[j], ub[j])) for j = 1:dim]
        cost[i] = objfun(bonobo[i, :])
    end

    bestcost, ID = findmin(cost)
    alphabonobo = bonobo[ID:ID, :]
    pbestcost = bestcost

    # Initialization of other parameters
    npc = 0
    ppc = 0
    p_xgm = p_xgm_initial
    tsgs_factor_initial = 0.5 * tsgs_factor_max
    tsgs_factor = tsgs_factor_initial
    p_p = 0.5
    p_d = 0.5
    it = 1

    # Main Loop of BO
    while it <= max_iter
        tsgs_max = max(2, ceil(Int, npop * tsgs_factor))

        for i = 1:npop
            newbonobo = zeros(1, dim)
            B = collect(1:npop)
            deleteat!(B, i)  # Exclude the current index

            tsg = rand(2:tsgs_max)

            q = rand(B, tsg)
            temp_cost = cost[q]
            _, ID1 = findmin(temp_cost)
            p = q[ID1]

            if cost[i] < cost[p]
                p = q[rand(1:tsg)]  # Randomly select p
                flag = 1
            else
                flag = -1
            end

            # Creation of newbonobo
            if rand() <= p_p
                r1 = rand(1, dim)
                newbonobo = bonobo[i:i, :] .+ scab * r1 .* (alphabonobo .- bonobo[i:i, :]) .+ flag * scsb * (1 .- r1) .* (bonobo[i:i, :] .- bonobo[p:p, :])
            else
                for j = 1:dim
                    if rand() <= p_xgm
                        rand_var = rand()

                        if alphabonobo[1, j] >= bonobo[i, j]
                            if rand() <= p_d
                                beta1 = exp(rand_var^2 + rand_var - 2 / rand_var)
                                newbonobo[1, j] = bonobo[i, j] + beta1 * (ub[j] - bonobo[i, j])
                            else
                                beta2 = exp(-rand_var^2 + 2 * rand_var - 2 / rand_var)
                                newbonobo[1, j] = bonobo[i, j] - beta2 * (bonobo[i, j] - lb[j])
                            end
                        else
                            if rand() <= p_d
                                beta1 = exp(rand_var^2 + rand_var - 2 / rand_var)
                                newbonobo[1, j] = bonobo[i, j] - beta1 * (bonobo[i, j] - lb[j])
                            else
                                beta2 = exp(-rand_var^2 + 2 * rand_var - 2 / rand_var)
                                newbonobo[1, j] = bonobo[i, j] + beta2 * (ub[j] - bonobo[i, j])
                            end
                        end
                    else
                        if (flag == 1) || (rand() <= p_d)
                            newbonobo[1, j] = bonobo[i, j] + flag * exp(-rand()) * (bonobo[i, j] - bonobo[p, j])
                        else
                            newbonobo[1, j] = bonobo[p, j]
                        end
                    end
                end
            end

            for j = 1:dim
                if newbonobo[1, j] > ub[j]
                    newbonobo[1, j] = ub[j]
                elseif newbonobo[1, j] < lb[j]
                    newbonobo[1, j] = lb[j]
                end
            end

            newcost = objfun(newbonobo[1, :])  # New cost evaluation

            if (newcost < cost[i]) || (rand() <= p_xgm)
                cost[i] = newcost
                bonobo[i, :] = newbonobo[1, :]  # Update bonobo

                if newcost < bestcost
                    bestcost = newcost
                    alphabonobo = newbonobo  # Update best solution
                end
            end
        end

        if bestcost < pbestcost
            npc = 0  # Positive phase
            ppc += 1
            cp = min(0.5, ppc * rcpp)
            pbestcost = bestcost
            p_xgm = p_xgm_initial
            p_p = 0.5 + cp
            p_d = p_p
            tsgs_factor = min(tsgs_factor_max, tsgs_factor_initial + ppc * (rcpp^2))
        else
            npc += 1
            ppc = 0
            cp = -(min(0.5, npc * rcpp))
            p_xgm = min(0.5, p_xgm_initial + npc * (rcpp^2))
            tsgs_factor = max(0, tsgs_factor_initial - npc * (rcpp^2))
            p_p = 0.5 + cp
            p_d = 0.5
        end

        convergence[it] = bestcost
        it += 1
    end

    # return bestcost, alphabonobo, convergence
    return OptimizationResult(
        vec(alphabonobo),
        bestcost,
        convergence)
end

function BO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return BO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end