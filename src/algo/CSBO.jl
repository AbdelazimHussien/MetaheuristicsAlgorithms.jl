struct Plant
    Position::Vector{Float64}
    Cost::Float64
end

"""
# References:

- Ghasemi, Mojtaba, Mohammad-Amin Akbari, Changhyun Jun, Sayed M. Bateni, Mohsen Zare, Amir Zahedi, Hao-Ting Pai, Shahab S. Band, Massoud Moslehpour, and Kwok-Wing Chau. 
  "Circulatory System Based Optimization (CSBO): an expert multilevel biologically inspired meta-heuristic algorithm." 
  Engineering Applications of Computational Fluid Mechanics 16, no. 1 (2022): 1483-1525.

"""
function CSBO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return CSBO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter) 
end

function CSBO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    MaxFE = npop * max_iter
    NR = div(npop, 3)
    BestSolPosition = zeros(dim)
    BestSolCost = Inf

    pop = Vector{Plant}(undef, npop)
    sigma1 = [rand(dim) for _ = 1:npop]

    position = initializationLogistic(npop, dim, ub, lb)
    for i = 1:npop
        # position = rand(lb:ub, dim)
        cost = objfun(position[i, :])
        pop[i] = Plant(position[i, :], cost)

        if cost <= BestSolCost
            BestSolPosition = position[i, :]
            BestSolCost = cost
        end
    end

    BestCost = Float64[BestSolCost]

    it = npop

    while it <= MaxFE
        Costs = [p.Cost for p in pop]
        BestCostNow = minimum(Costs)
        WorstCost = maximum(Costs)

        for i = 1:npop
            A = shuffle(1:npop)
            A = filter(x -> x != i, A)
            a1, a2, a3 = A[1:3]

            AA1 = (pop[a2].Cost - pop[a3].Cost) / abs(pop[a3].Cost - pop[a2].Cost)
            AA2 = (pop[a1].Cost - pop[i].Cost) / abs(pop[a1].Cost - pop[i].Cost)
            AA1 = (pop[a2].Cost == pop[a3].Cost) ? 1.0 : AA1
            AA2 = (pop[a1].Cost == pop[i].Cost) ? 1.0 : AA2

            pos = pop[i].Position .+ AA2 * sigma1[i] .* (pop[i].Position - pop[a1].Position) .+
                  AA1 * sigma1[i] .* (pop[a3].Position - pop[a2].Position)

            pos = clamp.(pos, lb, ub)
            cost = objfun(pos)

            if cost < pop[i].Cost
                pop[i] = Plant(pos, cost)
                if cost <= BestSolCost
                    BestSolPosition = pos
                    BestSolCost = cost
                end
            end

            it += 1
            push!(BestCost, BestSolCost)
        end

        for i = 1:(npop-NR)
            ratioi = (pop[i].Cost - WorstCost) / (BestCostNow - WorstCost)
            sigma1[i] = fill(ratioi, dim)
            A = shuffle(1:npop)
            A = filter(x -> x != i, A)
            a1, a2, a3 = A[1:3]

            pos = pop[a1].Position .+ sigma1[i] .* (pop[a3].Position - pop[a2].Position)
            pos = clamp.(pos, lb, ub)

            if rand() > 0.9
                pos .= pop[i].Position
            end

            cost = objfun(pos)
            if cost < pop[i].Cost
                pop[i] = Plant(pos, cost)
                if cost <= BestSolCost
                    BestSolPosition = pos
                    BestSolCost = cost
                end
            end

            it += 1
            push!(BestCost, BestSolCost)
        end

        for i = (npop-NR+1):npop
            pos = pop[i].Position .+ (randn() / it) .* randn(dim)
            pos = clamp.(pos, lb, ub)
            cost = objfun(pos)

            if cost < pop[i].Cost
                pop[i] = Plant(pos, cost)
                if cost <= BestSolCost
                    BestSolPosition = pos
                    BestSolCost = cost
                end
            end

            sigma1[i] = rand(dim)
            it += 1
            push!(BestCost, BestSolCost)
        end
    end
    # return BestSolCost, BestSolPosition, BestCost
    return OptimizationResult(
        BestSolPosition,
        BestSolCost,
        BestCost[1:npop:(max_iter-1)*npop+1])
end

function CSBO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return CSBO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end