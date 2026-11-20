"""
Wang, Liying, Haiping Du, Zhenxing Zhang, Gang Hu, Seyedali Mirjalili, Nima Khodadadi, Abdelazim G. Hussien, Yingying Liao, and Weiguo Zhao. 
"Tianji’s horse racing optimization (THRO): a new metaheuristic inspired by ancient wisdom and its engineering optimization applications." 
Artificial Intelligence Review 58, no. 9 (2025): 1-111.
"""
function THRO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    # FunIndex: Index of function
    # npop: Sum of horse population of Tianji and King
    # nPop: Number of horses each in Tianji's and King's population
    # max_iter: Maximum number of iterations

    nPop = div(npop, 2)
    # Low, ub, dim = FunRange(FunIndex)

    Tianji_PopPos = zeros(nPop, dim)
    Tianji_PopFit = zeros(nPop)
    King_PopPos = zeros(nPop, dim)
    King_PopFit = zeros(nPop)

    for i = 1:nPop
        Tianji_PopPos[i, :] = lb .+ rand(dim) .* (ub .- lb)
        # Tianji_PopFit[i] = BenFunctions(Tianji_PopPos[i, :], FunIndex, dim)
        Tianji_PopFit[i] = objfun(Tianji_PopPos[i, :])
        King_PopPos[i, :] = lb .+ rand(dim) .* (ub .- lb)
        King_PopFit[i] = objfun(King_PopPos[i, :])# BenFunctions(King_PopPos[i, :], FunIndex, dim)
    end

    BestFit = Inf
    BestX = zeros(dim)

    for i = 1:nPop
        if Tianji_PopFit[i] <= BestFit
            BestFit = Tianji_PopFit[i]
            BestX .= Tianji_PopPos[i, :]
        end
    end

    for i = 1:nPop
        if King_PopFit[i] <= BestFit
            BestFit = King_PopFit[i]
            BestX .= King_PopPos[i, :]
        end
    end

    His_BestFit = zeros(max_iter)

    for It = 1:max_iter
        Rand_npop = randperm(2 * nPop)
        PopPos = vcat(Tianji_PopPos, King_PopPos)
        PopFit = vcat(Tianji_PopFit, King_PopFit)

        Tianji_PopPos = PopPos[Rand_npop[1:nPop], :]
        Tianji_PopFit = PopFit[Rand_npop[1:nPop]]
        King_PopPos = PopPos[Rand_npop[nPop+1:end], :]
        King_PopFit = PopFit[Rand_npop[nPop+1:end]]

        Tianji_PopFit, ind = sort(Tianji_PopFit), sortperm(Tianji_PopFit)
        Tianji_PopPos = Tianji_PopPos[ind, :]

        King_PopFit, ind = sort(King_PopFit), sortperm(King_PopFit)
        King_PopPos = King_PopPos[ind, :]

        T_B = zeros(nPop, dim)
        K_B = zeros(nPop, dim)
        p = 1 - It / max_iter

        for i = 1:nPop
            Randdim = randperm(dim)
            RandNum = ceil(Int, sin(pi / 2 * rand()) * dim)
            T_B[i, Randdim[1:RandNum]] .= 1

            Randdim = randperm(dim)
            RandNum = ceil(Int, sin(pi / 2 * rand()) * dim)
            K_B[i, Randdim[1:RandNum]] .= 1
        end

        Tianji_SlowestId = nPop
        Tianji_FastestId = 1
        King_SlowestId = nPop
        King_FastestId = 1

        for i = 1:nPop
            Tianji_Alpha = 1 + round(Int, 0.5 * (0.5 + rand())) * randn()
            T_Beta = round(Int, 0.5 * (0.1 + rand())) * randn()
            King_Alpha = 1 + round(Int, 0.5 * (0.5 + rand())) * randn()
            K_Beta = round(Int, 0.5 * (0.1 + rand())) * randn()
            Tianji_R = levy(1) .* T_B[i, :]
            King_R = levy(1) .* K_B[i, :]

            # Scenario 1
            if Tianji_PopFit[Tianji_SlowestId] < King_PopFit[King_SlowestId]
                Tianji_SlowestPos = Tianji_PopPos[Tianji_SlowestId, :]
                King_SlowestPos = King_PopPos[King_SlowestId, :]
                Tianji_newPopPos =
                    (
                        (p .* Tianji_SlowestPos .+ (1 - p) .* Tianji_PopPos[1, :]) .+
                        Tianji_R .*
                        (Tianji_PopPos[1, :] .- Tianji_SlowestPos .+ p .* (vec(mean(Tianji_PopPos, dims=1)) .- vec(mean(King_PopPos, dims=1))))
                    ) .* Tianji_Alpha .+ T_Beta
                # Tianji_newPopPos = SpaceBound(Tianji_newPopPos, ub, lb)
                Tianji_newPopPos = clamp.(Tianji_newPopPos, lb, ub)
                # Tianji_newPopFit = BenFunctions(Tianji_newPopPos, FunIndex, dim)
                Tianji_newPopFit = objfun(Tianji_newPopPos)
                if Tianji_newPopFit < Tianji_PopFit[Tianji_SlowestId]
                    Tianji_PopFit[Tianji_SlowestId] = Tianji_newPopFit
                    Tianji_PopPos[Tianji_SlowestId, :] = Tianji_newPopPos
                end

                King_newPopPos =
                    ((p .* King_SlowestPos .+ (1 - p) .* Tianji_SlowestPos) .+ King_R .*
                                                                               (Tianji_SlowestPos .- King_SlowestPos .+ p .* (vec(mean(Tianji_PopPos, dims=1)) .- vec(mean(King_PopPos, dims=1))))) .*
                    King_Alpha .+ K_Beta
                # King_newPopPos = SpaceBound(King_newPopPos, ub, lb)
                King_newPopPos = clamp.(King_newPopPos, lb, ub)
                # King_newPopFit = BenFunctions(King_newPopPos, FunIndex, dim)
                King_newPopFit = objfun(King_newPopPos)
                if King_newPopFit < King_PopFit[King_SlowestId]
                    King_PopFit[King_SlowestId] = King_newPopFit
                    King_PopPos[King_SlowestId, :] = King_newPopPos
                end

                Tianji_SlowestId -= 1
                King_SlowestId -= 1

            else
                # Scenario 2
                if Tianji_PopFit[Tianji_SlowestId] > King_PopFit[King_SlowestId]
                    Tianji_SlowestPos = Tianji_PopPos[Tianji_SlowestId, :]
                    King_FastestPos = King_PopPos[King_FastestId, :]
                    Tr1 = rand(1:nPop)
                    Tianji_newPopPos =
                        (
                            (p .* Tianji_SlowestPos .+ (1 - p) .* Tianji_PopPos[Tr1, :]) .+
                            Tianji_R .*
                            (Tianji_PopPos[Tr1, :] .- Tianji_SlowestPos .+ p .* (vec(mean(Tianji_PopPos, dims=1)) .- vec(mean(King_PopPos, dims=1))))
                        ) .* Tianji_Alpha .+ T_Beta
                    # Tianji_newPopPos = SpaceBound(Tianji_newPopPos, ub, lb)
                    Tianji_newPopPos = clamp.(Tianji_newPopPos, lb, ub)
                    # Tianji_newPopFit = BenFunctions(Tianji_newPopPos, FunIndex, dim)
                    println(size(Tianji_newPopPos))
                    Tianji_newPopFit = objfun(Tianji_newPopPos)
                    if Tianji_newPopFit < Tianji_PopFit[Tianji_SlowestId]
                        Tianji_PopFit[Tianji_SlowestId] = Tianji_newPopFit
                        Tianji_PopPos[Tianji_SlowestId, :] = Tianji_newPopPos
                    end

                    King_newPopPos =
                        (
                            (p .* King_FastestPos .+ (1 - p) .* King_PopPos[1, :]) .+
                            King_R .*
                            (King_PopPos[1, :] .- King_FastestPos .+ p .* (vec(mean(Tianji_PopPos, dims=1)) .- vec(mean(King_PopPos, dims=1))))
                        ) .* King_Alpha .+ K_Beta
                    # King_newPopPos = SpaceBound(King_newPopPos, ub, lb)
                    King_newPopPos = clamp.(King_newPopPos, lb, ub)
                    # King_newPopFit = BenFunctions(King_newPopPos, FunIndex, dim)
                    King_newPopFit = objfun(King_newPopPos)
                    if King_newPopFit < King_PopFit[King_FastestId]
                        King_PopFit[King_FastestId] = King_newPopFit
                        King_PopPos[King_FastestId, :] = King_newPopPos
                    end

                    Tianji_SlowestId -= 1
                    King_FastestId += 1

                else
                    # Scenario 3
                    if Tianji_PopFit[Tianji_FastestId] < King_PopFit[King_FastestId]
                        Tianji_FastestPos = Tianji_PopPos[Tianji_FastestId, :]
                        King_FastestPos = King_PopPos[King_FastestId, :]
                        Tianji_newPopPos =
                            (
                                (p .* Tianji_FastestPos .+ (1 - p) .* Tianji_PopPos[1, :]) .+
                                Tianji_R .*
                                (Tianji_PopPos[1, :] .- Tianji_FastestPos .+ p .* (vec(mean(Tianji_PopPos, dims=1)) .- vec(mean(King_PopPos, dims=1))))
                            ) .* Tianji_Alpha .+ T_Beta
                        # Tianji_newPopPos = SpaceBound(Tianji_newPopPos, ub, lb)
                        Tianji_newPopPos = clamp.(Tianji_newPopPos, lb, ub)
                        # Tianji_newPopFit = BenFunctions(Tianji_newPopPos, FunIndex, dim)
                        Tianji_newPopFit = objfun(Tianji_newPopPos)
                        if Tianji_newPopFit < Tianji_PopFit[Tianji_FastestId]
                            Tianji_PopFit[Tianji_FastestId] = Tianji_newPopFit
                            Tianji_PopPos[Tianji_FastestId, :] = Tianji_newPopPos
                        end

                        King_newPopPos =
                            (
                                (p .* King_FastestPos .+ (1 - p) .* King_PopPos[1, :]) .+
                                King_R .*
                                (King_PopPos[1, :] .- King_FastestPos .+ p .* (vec(mean(Tianji_PopPos, dims=1)) .- vec(mean(King_PopPos, dims=1))))
                            ) .* King_Alpha .+ K_Beta
                        # King_newPopPos = SpaceBound(King_newPopPos, ub, lb)
                        King_newPopPos = clamp.(King_newPopPos, lb, ub)
                        # King_newPopFit = BenFunctions(King_newPopPos, FunIndex, dim)
                        King_newPopFit = objfun(King_newPopPos)
                        if King_newPopFit < King_PopFit[King_FastestId]
                            King_PopFit[King_FastestId] = King_newPopFit
                            King_PopPos[King_FastestId, :] = King_newPopPos
                        end

                        Tianji_FastestId += 1
                        King_FastestId += 1
                    else
                        # Scenario 4
                        if Tianji_PopFit[Tianji_FastestId] > King_PopFit[King_FastestId]
                            Tianji_FastestPos = Tianji_PopPos[Tianji_FastestId, :]
                            King_SlowestPos = King_PopPos[King_SlowestId, :]
                            King_newPopPos =
                                (
                                    (p .* King_SlowestPos .+ (1 - p) .* King_PopPos[1, :]) .+
                                    King_R .*
                                    (King_PopPos[1, :] .- King_SlowestPos .+ p .* (vec(mean(Tianji_PopPos, dims=1)) .- vec(mean(King_PopPos, dims=1))))
                                ) .* King_Alpha .+ K_Beta
                            # King_newPopPos = SpaceBound(King_newPopPos, ub, lb)
                            King_newPopPos = clamp.(King_newPopPos, lb, ub)
                            # King_newPopFit = BenFunctions(King_newPopPos, FunIndex, dim)
                            King_newPopFit = objfun(King_newPopPos)
                            if King_newPopFit < King_PopFit[King_SlowestId]
                                King_PopFit[King_SlowestId] = King_newPopFit
                                King_PopPos[King_SlowestId, :] = King_newPopPos
                            end

                            Tianji_newPopPos =
                                (
                                    (p .* Tianji_FastestPos .+ (1 - p) .* Tianji_PopPos[1, :]) .+
                                    Tianji_R .*
                                    (Tianji_PopPos[1, :] .- Tianji_FastestPos .+ p .* (vec(mean(Tianji_PopPos, dims=1)) .- vec(mean(King_PopPos, dims=1))))
                                ) .* Tianji_Alpha .+ T_Beta
                            # Tianji_newPopPos = SpaceBound(Tianji_newPopPos, ub, lb)
                            Tianji_newPopPos = clamp.(Tianji_newPopPos, lb, ub)
                            # Tianji_newPopFit = BenFunctions(Tianji_newPopPos, FunIndex, dim)
                            Tianji_newPopFit = objfun(Tianji_newPopPos)
                            if Tianji_newPopFit < Tianji_PopFit[Tianji_FastestId]
                                Tianji_PopFit[Tianji_FastestId] = Tianji_newPopFit
                                Tianji_PopPos[Tianji_FastestId, :] = Tianji_newPopPos
                            end

                            Tianji_FastestId += 1
                            King_SlowestId -= 1
                        end
                    end
                end
            end
        end

        # Update best solution
        for i = 1:nPop
            if Tianji_PopFit[i] <= BestFit
                BestFit = Tianji_PopFit[i]
                BestX .= Tianji_PopPos[i, :]
            end
        end
        for i = 1:nPop
            if King_PopFit[i] <= BestFit
                BestFit = King_PopFit[i]
                BestX .= King_PopPos[i, :]
            end
        end

        His_BestFit[It] = BestFit
    end

    # return BestFit, BestX, His_BestFit
    return OptimizationResult(
        BestX,
        BestFit,
        His_BestFit)
end

function THRO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return THRO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function THRO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return THRO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

# function Levy(d)
#     b = 1.5
#     s = (gamma(1 + b) * sin(pi * b / 2) / (gamma((1 + b) / 2) * b * 2^((b - 1) / 2)))^(1 / b)
#     u = randn(d) * s
#     v = randn(d)
#     sigma = u ./ abs.(v) .^ (1 / b)
#     return sigma
# end
