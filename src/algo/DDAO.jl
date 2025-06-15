"""
# References:

- Ghafil, H. N., & Jármai, K. (2020). 
    Dynamic differential annealed optimization: New metaheuristic optimization algorithm for engineering applications. 
    Applied Soft Computing, 93, 106392.
    
"""
function DDAO(Npop, MaxIt, L_limit, U_limit, Nvar, CostFunction)
    VarLength = Nvar       
    
    MaxSubIt = 1000    
    T0 = 2000.0        
    alpha = 0.995      
    
    empty_template = Dict("Phase" => [], "Cost" => Inf)
    pop = [deepcopy(empty_template) for _ in 1:Npop]

    BestSol = Dict("Phase" => [], "Cost" => Inf)


    for i in 1:Npop
        pop[i]["Phase"] = L_limit .+ rand(VarLength) .* (U_limit - L_limit)
        
        pop[i]["Cost"] = CostFunction(pop[i]["Phase"])

        if pop[i]["Cost"] <= BestSol["Cost"]
            BestSol = deepcopy(pop[i])
        end
    end


    BestCost = zeros(MaxIt)

    T = T0

    for t in 1:MaxIt
        newpop = [deepcopy(empty_template) for _ in 1:MaxSubIt]

        for subit in 1:MaxSubIt
            newpop[subit]["Phase"] = L_limit .+ rand(VarLength) .* (U_limit - L_limit)
            newpop[subit]["Phase"] = clamp.(newpop[subit]["Phase"], L_limit, U_limit)
            newpop[subit]["Cost"] = CostFunction(newpop[subit]["Phase"])
        end

        SortOrder = sortperm([newpop[subit]["Cost"] for subit in 1:MaxSubIt])
        newpop = newpop[SortOrder]
        bnew = newpop[1]
        kk = rand(1:Npop)
        bb = rand(1:Npop)

        Mnew = deepcopy(empty_template)
        if isodd(t)
            Mnew["Phase"] = (pop[kk]["Phase"] - pop[bb]["Phase"]) + bnew["Phase"]
        else
            Mnew["Phase"] = (pop[kk]["Phase"] - pop[bb]["Phase"]) + bnew["Phase"] * rand()
        end

        Mnew["Phase"] = clamp.(Mnew["Phase"], L_limit, U_limit)

        Mnew["Cost"] = CostFunction(Mnew["Phase"])

        for i in 1:Npop
            if Mnew["Cost"] <= pop[i]["Cost"]
                pop[i] = deepcopy(Mnew)
            else
                DELTA = Mnew["Cost"] - pop[i]["Cost"]
                P = exp(-DELTA / T)
                if rand() <= P
                    pop[end] = deepcopy(Mnew)
                end
            end

            if pop[i]["Cost"] <= BestSol["Cost"]
                BestSol = deepcopy(pop[i])
            end
        end

        
        T *= alpha
    end

    y = BestSol["Cost"]

    return  BestSol["Cost"], BestSol["Phase"], BestCost
end