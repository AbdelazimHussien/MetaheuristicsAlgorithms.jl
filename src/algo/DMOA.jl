mutable struct Mongoose
    Position::Vector{Float64}
    Cost::Float64
end

"""
# References:

- Agushaka, Jeffrey O., Absalom E. Ezugwu, and Laith Abualigah. 
    "Dwarf mongoose optimization algorithm." 
    Computer methods in applied mechanics and engineering 391 (2022): 114570.
    
"""
function DMOA(nPop, MaxIt, VarMin, VarMax, nVar, F_obj)
    VarSize = nVar  

    nBabysitter = 3  
    nAlphaGroup = nPop - nBabysitter  
    nScout = nAlphaGroup  

    L = round(Int, 0.6 * nVar * nBabysitter)  
    peep = 2.0  

    pop = [Mongoose(zeros(VarSize), Inf) for _ in 1:nAlphaGroup]

    BestSol = Mongoose(zeros(VarSize), Inf)
    tau = Inf
    sm = fill(Inf, nAlphaGroup)  

    for i in 1:nAlphaGroup
        pop[i].Position = rand(VarSize) .* (VarMax .- VarMin) .+ VarMin
        pop[i].Cost = F_obj(pop[i].Position)
        if pop[i].Cost <= BestSol.Cost
            BestSol = pop[i]
        end
    end

    C = zeros(Int, nAlphaGroup)
    CF = (1 - 1 / MaxIt)^(2 * 1 / MaxIt)

    BestCost = zeros(Float64, MaxIt)

    for it in 1:MaxIt
        F = zeros(Float64, nAlphaGroup)
        MeanCost = mean([pop[i].Cost for i in 1:nAlphaGroup])
        
        for i in 1:nAlphaGroup
            F[i] = exp(-pop[i].Cost / MeanCost)  
        end
        
        P = F / sum(F)  
        
        for m in 1:nAlphaGroup
            i = RouletteWheelSelection(P)
            
            K = setdiff(1:nAlphaGroup, i)
            k = K[rand(1:length(K))]
            
            phi = (peep / 2) * rand(VarSize) .* (2 .* rand(VarSize) .- 1)
            
            newpop = Mongoose(zeros(VarSize), Inf)
            newpop.Position = pop[i].Position .+ phi .* (pop[i].Position .- pop[k].Position)
            
            newpop.Cost = F_obj(newpop.Position)
            
            if newpop.Cost <= pop[i].Cost
                pop[i] = newpop
            else
                C[i] += 1
            end
        end   

        for i in 1:nScout
            K = setdiff(1:nAlphaGroup, i)
            k = K[rand(1:length(K))]
            
            phi = (peep / 2) * rand(VarSize) .* (2 .* rand(VarSize) .- 1)
            
            newpop = Mongoose(zeros(VarSize), Inf)
            newpop.Position = pop[i].Position .+ phi .* (pop[i].Position .- pop[k].Position)
            
            newpop.Cost = F_obj(newpop.Position)
            
            sm[i] = (newpop.Cost - pop[i].Cost) / max(newpop.Cost, pop[i].Cost)
            
            if newpop.Cost <= pop[i].Cost
                pop[i] = newpop
            else
                C[i] += 1
            end
        end    

        for i in 1:nBabysitter
            if C[i] >= L
                pop[i].Position = rand(VarSize) .* (VarMax .- VarMin) .+ VarMin
                pop[i].Cost = F_obj(pop[i].Position)
                C[i] = 0
            end
        end    

        for i in 1:nAlphaGroup
            if pop[i].Cost <= BestSol.Cost
                BestSol = pop[i]
            end
        end    
        
        newtau = mean(sm)
        for i in 1:nScout
            M = (pop[i].Position .* sm[i]) ./ pop[i].Position
            if newtau > tau
                newpop = Mongoose(zeros(VarSize), Inf)
                newpop.Position = pop[i].Position .- CF * rand(VarSize) .* (pop[i].Position .- M)
            else
                newpop = Mongoose(zeros(VarSize), Inf)
                newpop.Position = pop[i].Position .+ CF * rand(VarSize) .* (pop[i].Position .- M)
            end
            tau = newtau
        end
       
        for i in 1:nAlphaGroup
            if pop[i].Cost <= BestSol.Cost
                BestSol = pop[i]
            end
        end
    
        BestCost[it] = BestSol.Cost
        
    end

    return BestSol.Cost, BestSol.Position, BestCost
end