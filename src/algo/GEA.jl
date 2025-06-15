"""
Ghasemi, Mojtaba, Mohsen Zare, Amir Zahedi, Mohammad-Amin Akbari, Seyedali Mirjalili, and Laith Abualigah. 
"Geyser inspired algorithm: a new geological-inspired meta-heuristic for real-parameter and constrained engineering optimization." 
Journal of Bionic Engineering 21, no. 1 (2024): 374-408.
"""


function GEA(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction)
    VarSize = (1, nVar)  
    Nc = Int(floor(nPop / 3))  
    FEs = nPop * MaxIt 
    
    Positions = rand(nPop, nVar) * (VarMax - VarMin) .+ VarMin
    Costs = zeros(Float64, nPop)
    for i in 1:nPop
        Costs[i] = CostFunction(Positions[i,:])  
    end
    BestCost = minimum(Costs) 
    BestPosition = Positions[argmin(Costs)]  
    BestCosts = zeros(FEs)  

    it = nPop

    while it < FEs
        for i in 1:nPop
            D1 = [sum(Positions[i,:] .* Positions[j,:]) / sqrt(sum(Positions[i,:] .^ 2) * sum(Positions[j,:] .^ 2)) for j in 1:nPop]
            D1[i] = Inf  
            S1 = minimum(D1)
            j1 = argmin(D1)

            CS = sort(Costs)
            dif = CS[end] - CS[1]  
            G = (Costs[i] - CS[1]) / dif  
            G = dif == 0 ? 0 : G  

            P_i = sqrt(((G^(2 / it)) - (G^((it + 1) / it))) * (it / (it - 1)))  

            ImpFitness = sum(Costs[1:Nc])
            p = [Costs[ii] / ImpFitness for ii in 1:Nc]
            ImpFitness = ImpFitness == 0 ? 1e-320 : ImpFitness

            i1 = sum(p) == 0 ? 1 : RouletteWheelSelectionFun(p)

            flag = 0
            newsol_Position = Positions[j1,:] .+ rand(nVar) .* (Positions[i1,:] - Positions[j1,:]) .+ rand(nVar) .* (Positions[i1,:] - Positions[i,:])
            newsol_Position = max.(newsol_Position, VarMin)
            newsol_Position = min.(newsol_Position, VarMax)
            newsol_Cost = CostFunction(newsol_Position)
            it += 1

            if it > FEs  
                break
            end

            if newsol_Cost <= Costs[i]
                Positions[i,:] = newsol_Position
                Costs[i] = newsol_Cost
            end
            if newsol_Cost <= BestCost
                BestCost = newsol_Cost
                BestPosition = newsol_Position
            end
            BestCosts[it] = BestCost

            if flag == 0
                i2 = RouletteWheelSelectionFun(1 .- p)  
                newsol_Position = Positions[i2,:] .+ rand() * (P_i - rand()) .* rand(nVar) * (VarMax - VarMin) .+ VarMin
                newsol_Position = max.(newsol_Position, VarMin)
                newsol_Position = min.(newsol_Position, VarMax)
                newsol_Cost = CostFunction(newsol_Position)
                it += 1

                if newsol_Cost <= Costs[i]
                    Positions[i,:] = newsol_Position
                    Costs[i] = newsol_Cost
                end
                if newsol_Cost <= BestCost
                    BestCost = newsol_Cost
                    BestPosition = newsol_Position
                end
                BestCosts[it] = BestCost
            end
        end

        sorted_indices = sortperm(Costs)
        Positions = Positions[sorted_indices,:]
        Costs = Costs[sorted_indices]
    end
    return BestCost, BestPosition, BestCosts
end

function RouletteWheelSelectionFun(p)
    c = cumsum(p)           
    r = rand()              
    i = findfirst(x -> x >= r, c)  
    return i                
end