mutable struct Individual
    Position::Vector{Float64}
    Cost::Float64
end

"""
# References:

- Akbari, Mohammad Amin, Mohsen Zare, Rasoul Azizipanah-Abarghooee, Seyedali Mirjalili, and Mohamed Deriche. "The cheetah optimizer: A nature-inspired metaheuristic algorithm for large-scale optimization problems." Scientific reports 12, no. 1 (2022): 10953.

"""
function CO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return CO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter) 
end

function CO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    BestSol = Individual([], Inf)
    pop = [Individual([], Inf) for _ = 1:npop]

    MaxEEs = npop * max_iter

    xd = zeros(dim)

    if length(lb) == 1
        ub = ub .* ones(dim)
        lb = lb .* ones(dim)
    end

    for i = 1:npop
        # pop[i].Position = lb .+ rand(dim) .* (ub .- lb)
        pop[i].Position = vec(initialization(1, dim, ub, lb))
        pop[i].Cost = objfun(pop[i].Position)
        if pop[i].Cost < BestSol.Cost
            BestSol = deepcopy(pop[i])
        end
    end

    pop1 = copy(pop)
    BestCost = Float64[]
    X_best = deepcopy(BestSol)
    Globest = Float64[]

    t = 0
    it = 1
    T = ceil(Int, dim / 10) * 60
    FEs = 0
    i = 0
    his = zeros(max_iter)
    nhis = 0

    while FEs <= MaxEEs
        m = rand(2:npop)
        i0 = rand(1:npop, m)
        for k = 1:m
            i = rand(1:npop)
            i = i0[k]

            if k == length(i0) && k != 1
                a = i0[k-1]
            else
                a = i0[k+1]
            end

            X = pop[i].Position
            X1 = pop[a].Position
            Xb = BestSol.Position
            Xbest = X_best.Position

            kk = 0
            if i <= 2 && t > 2 && t > Int(ceil(0.2 * T + 1)) && abs(BestCost[t-2] - BestCost[t-Int(ceil(0.2 * T + 1))]) <= 0.0001 * Globest[t-1]
                X = X_best.Position
                kk = 0
            elseif i == 3
                X = BestSol.Position
                kk = -0.1 * rand() * t / T
            else
                kk = 0.25
            end

            if it % 100 == 0 || it == 1
                xd = randperm(dim)
            end

            Z = copy(X)

            for j in xd
                r_Hat = randn()
                r1 = rand()
                alpha = (k == 1) ? 0.0001 * t / T * (ub[j] - lb[j]) : 0.0001 * t / T * abs(Xb[j] - X[j]) + 0.001 * round(rand() > 0.9)

                r = randn()
                r_Check = abs(r)^exp(r / 2) * sin(2 * π * r)
                beta = X1[j] - X[j]

                h0 = exp(2 - 2 * t / T)
                H = abs(2 * r1 * h0 - h0)

                r2 = rand()
                r3 = kk + rand()

                if r2 <= r3
                    r4 = 3 * rand()
                    if H > r4
                        Z[j] = X[j] + r_Hat^-1 * alpha
                    else
                        Z[j] = Xbest[j] + r_Check * beta
                    end
                else
                    Z[j] = X[j]
                end
            end

            Z = clamp.(Z, lb, ub)

            NewSol = Individual(Z, objfun(Z))
            if NewSol.Cost < pop[i].Cost
                pop[i] = NewSol
                if pop[i].Cost < BestSol.Cost
                    BestSol = deepcopy(pop[i])
                end
            end
            FEs += 1
        end

        t += 1

        if t > T && t - round(T) - 1 >= 1 && t > 2
            if abs(BestCost[t-1] - BestCost[t-round(T)-1]) <= abs(0.01 * BestCost[t-1])
                best = copy(X_best.Position)
                j0 = rand(1:dim, ceil(Int, dim / 10 * rand()))
                best[j0] = lb[j0] .+ rand(length(j0)) .* (ub[j0] .- lb[j0])
                BestSol.Cost = objfun(best)
                BestSol.Position = best
                FEs += 1

                i0 = rand(1:npop, round(1 * npop))
                pop[i0[npop-m+1:npop]] = pop1[i0[1:m]]

                pop[i] = X_best
                t = 1
            end
        end

        it += 1

        if BestSol.Cost < X_best.Cost
            X_best = deepcopy(BestSol)
        end
        t > length(BestCost) ? push!(BestCost, BestSol.Cost) : (BestCost[t] = BestSol.Cost)
        t > length(Globest) ? push!(Globest, X_best.Cost) : (Globest[t] = X_best.Cost)
        while nhis < min(max_iter, FEs ÷ npop)
            nhis += 1
            his[nhis] = X_best.Cost
        end
    end

    # return X_best.Cost, X_best.Position, BestCost
    return OptimizationResult(
        X_best.Position,
        X_best.Cost,
        his)
end

function CO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return CO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end