function exchange(Stallion)
    nStallion = length(Stallion)

    for i = 1:nStallion
        group_costs = [Stallion[i]["group"][j]["cost"] for j = 1:length(Stallion[i]["group"])]
        idx = argmin(group_costs)
        best_group = Stallion[i]["group"][idx]

        if best_group["cost"] < Stallion[i]["cost"]
            Stallion[i]["group"][idx]["pos"], Stallion[i]["pos"] = Stallion[i]["pos"], best_group["pos"]
            Stallion[i]["group"][idx]["cost"], Stallion[i]["cost"] = Stallion[i]["cost"], best_group["cost"]
        end
    end

    return Stallion
end

"""
# References:

- Naruei, Iraj, and Farshid Keynia.
  "Wild horse optimizer: A new meta-heuristic algorithm for solving engineering optimization problems."
  Engineering with computers 38, no. Suppl 4 (2022): 3025-3056.
"""
function WHO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return WHO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function WHO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    if length(lb) == 1 && length(ub) == 1
        lb = fill(lb, dim)
        ub = fill(ub, dim)
    end

    PS = 0.2
    PC = 0.13
    NStallion = ceil(Int, PS * npop)
    Nfoal = npop - NStallion
    r1 = r2 = r3 = rr = 0

    convergence_curve = zeros(max_iter)
    gBest = zeros(dim)
    gBestScore = Inf

    group = [Dict("pos" => lb .+ rand(dim) .* (ub .- lb), "cost" => objfun(lb .+ rand(dim) .* (ub .- lb))) for _ = 1:Nfoal]

    Stallion = [Dict("pos" => lb .+ rand(dim) .* (ub .- lb), "cost" => objfun(lb .+ rand(dim) .* (ub .- lb)), "group" => []) for _ = 1:NStallion]

    group = shuffle(group)
    i = 0
    k = 1
    for j = 1:Nfoal
        i += 1
        push!(Stallion[i]["group"], group[j])
        if i == NStallion
            i = 0
            k += 1
        end
    end

    Stallion = exchange(Stallion)
    idx = argmin([s["cost"] for s in Stallion])
    WH = Stallion[idx]
    gBest = WH["pos"]
    gBestScore = WH["cost"]
    convergence_curve[1] = WH["cost"]

    l = 2
    while l <= max_iter
        TDR = 1 - l / max_iter

        for i = 1:NStallion
            ngroup = length(Stallion[i]["group"])
            group_costs = [Stallion[i]["group"][j]["cost"] for j = 1:ngroup]
            sorted_idx = sortperm(group_costs)
            Stallion[i]["group"] = Stallion[i]["group"][sorted_idx]

            for j = 1:ngroup
                if rand() > PC
                    z = rand(dim) .< TDR
                    r1 = rand()
                    r2 = rand(dim)
                    idx = (z .== 0)
                    r3 = r1 .* idx .+ r2 .* .!idx

                    rr = -2 .+ 4 .* r3

                    Stallion[i]["group"][j]["pos"] = 2 .* r3 .* cos.(2 .* π .* rr) .* (Stallion[i]["pos"] .- Stallion[i]["group"][j]["pos"]) .+ Stallion[i]["pos"]
                else
                    A = randperm(NStallion)
                    A = setdiff(A, [i])
                    a, c = A[1], A[2]
                    x1 = Stallion[c]["group"][end]["pos"]
                    x2 = Stallion[a]["group"][end]["pos"]
                    Stallion[i]["group"][j]["pos"] = (x1 + x2) / 2
                end

                Stallion[i]["group"][j]["pos"] = clamp.(Stallion[i]["group"][j]["pos"], lb, ub)
                Stallion[i]["group"][j]["cost"] = objfun(Stallion[i]["group"][j]["pos"])
            end

            R = rand()
            if R < 0.5
                k = 2 .* r3 .* cos.(2 .* π .* rr) .* (WH["pos"] .- Stallion[i]["pos"]) .+ WH["pos"]
            else
                k = 2 .* r3 .* cos.(2 .* π .* rr) .* (WH["pos"] .- Stallion[i]["pos"]) .- WH["pos"]
            end

            k = clamp.(k, lb, ub)
            fk = objfun(k)
            if fk < Stallion[i]["cost"]
                Stallion[i]["pos"] = k
                Stallion[i]["cost"] = fk
            end
        end

        Stallion = exchange(Stallion)
        idx = argmin([s["cost"] for s in Stallion])
        if Stallion[idx]["cost"] < WH["cost"]
            WH = Stallion[idx]
        end

        gBest = WH["pos"]
        gBestScore = WH["cost"]
        convergence_curve[l] = WH["cost"]

        l += 1
    end

    # return gBestScore, gBest, convergence_curve
    return OptimizationResult(
        gBest,
        gBestScore,
        convergence_curve)
end

function WHO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)
    return WHO(npop, max_iter, problem.lb, problem.ub, problem.dim, problem.objfun)
end