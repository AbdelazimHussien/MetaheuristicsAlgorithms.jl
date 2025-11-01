"""
# References:

- Zhao, Weiguo, Liying Wang, and Zhenxing Zhang.
  "Supply-demand-based optimization: A novel economics-inspired algorithm for global optimization."
  Ieee Access 7 (2019): 73182-73206.
"""
function SupplyDO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return SupplyDO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function SupplyDO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, marketsize::Integer, max_iter::Integer)
    dim = length(lb)
    OneMarket = Dict(
        "CommPrice" => zeros(dim),
        "CommPriceFit" => Inf,
        "CommQuantity" => zeros(dim),
        "CommQuantityFit" => Inf,
    )

    market = fill(OneMarket, marketsize)
    best_f = Inf
    best_x = []

    for i = 1:marketsize
        # market[i]["CommPrice"] = rand(dim) .* (ub - lb) .+ lb
        market[i]["CommPrice"] = initialization(1, dim, lb, ub)
        market[i]["CommPriceFit"] = objfun(market[i]["CommPrice"])
        market[i]["CommQuantity"] = rand(dim) .* (ub - lb) .+ lb
        market[i]["CommQuantityFit"] = objfun(market[i]["CommQuantity"])

        if market[i]["CommQuantityFit"] <= market[i]["CommPriceFit"]
            market[i]["CommPriceFit"] = market[i]["CommQuantityFit"]
            market[i]["CommPrice"] = market[i]["CommQuantity"]
        end
    end

    for i = 1:marketsize
        if market[i]["CommPriceFit"] <= best_f
            best_x = market[i]["CommPrice"]
            best_f = market[i]["CommPriceFit"]
        end
    end

    his_best_fit = zeros(max_iter)

    for Iter = 1:max_iter
        a = 2 * (max_iter - Iter + 1) / max_iter
        F = zeros(marketsize)

        MeanQuantityFit = mean([market[i]["CommQuantityFit"] for i = 1:marketsize])
        for i = 1:marketsize
            F[i] = abs(market[i]["CommQuantityFit"] - MeanQuantityFit) + 1e-15
        end
        FQ = F / sum(F)

        MeanPriceFit = mean([market[i]["CommPriceFit"] for i = 1:marketsize])
        for i = 1:marketsize
            F[i] = abs(market[i]["CommPriceFit"] - MeanPriceFit) + 1e-15
        end
        FP = F / sum(F)

        MeanPrice = mean(hcat([market[i]["CommPrice"] for i = 1:marketsize]...), dims=2)'

        for i = 1:marketsize
            Ind = rand() < 0.5 ? 1 : 2
            k = findfirst(x -> x >= rand(), cumsum(FQ))
            CommQuantityEqu = market[k]["CommQuantity"]

            Alpha = a * sin.(2 * pi * rand(dim))
            Beta = 2 * cos.(2 * pi * rand(dim))

            CommPriceEqu = ifelse(rand() > 0.5, rand() * MeanPrice, market[findfirst(x -> x >= rand(), cumsum(FP))]["CommPrice"])

            NewCommQuantity = CommQuantityEqu .+ Alpha .* (market[i]["CommPrice"] .- vec(CommPriceEqu))

            NewCommQuantity .= clamp.(NewCommQuantity, lb, ub)

            NewCommQuantityFit = objfun(NewCommQuantity)

            if NewCommQuantityFit <= market[i]["CommQuantityFit"]
                market[i]["CommQuantityFit"] = NewCommQuantityFit
                market[i]["CommQuantity"] = NewCommQuantity
            end

            NewCommPrice = vec(CommPriceEqu) - Beta .* (NewCommQuantity .- CommQuantityEqu)
            NewCommPrice .= clamp.(NewCommPrice, lb, ub)
            NewCommPriceFit = objfun(NewCommPrice)

            if NewCommPriceFit <= market[i]["CommPriceFit"]
                market[i]["CommPriceFit"] = NewCommPriceFit
                market[i]["CommPrice"] = NewCommPrice
            end
        end

        for i = 1:marketsize
            if market[i]["CommQuantityFit"] <= market[i]["CommPriceFit"]
                market[i]["CommPriceFit"] = market[i]["CommQuantityFit"]
                market[i]["CommPrice"] = market[i]["CommQuantity"]
            end

            if market[i]["CommPriceFit"] <= best_f
                best_x = market[i]["CommPrice"]
                best_f = market[i]["CommPriceFit"]
            end
        end

        his_best_fit[Iter] = best_f
    end

    # return best_f, best_x, his_best_fit
    return OptimizationResult(
        best_x,
        best_f,
        his_best_fit)
end

function SupplyDO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return SupplyDO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end