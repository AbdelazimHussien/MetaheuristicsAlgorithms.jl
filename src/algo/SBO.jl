"""
Moosavi, Seyyed Hamid Samareh, and Vahid Khatibi Bardsiri. 
"Satin bowerbird optimizer: A new optimization algorithm to optimize ANFIS for software development effort estimation." 
Engineering Applications of Artificial Intelligence 60 (2017): 1-15.
"""


function SBO(nPop, MaxIt, lowerbound, upperbound, numbervar, costfcn)
    alpha = 0.94

    pMutation = 0.05
    
    Z = 0.02
    sigma=Z*(max(upperbound)-min(lowerbound))

    pop = [Dict("Position" => zeros(numbervar), "Cost" => Inf) for _ in 1:nPop]

    for i in 1:nPop
        pop[i]["Position"] = rand(numbervar) .* (upperbound - lowerbound) .+ lowerbound
        pop[i]["Cost"] = costfcn(pop[i]["Position"])
    end

    pop = sort(pop, by = x -> x["Cost"])

    BestSol = pop[1]
    elite = BestSol["Position"]

    BestCost = zeros(MaxIt)


    for it in 1:MaxIt
        newpop = deepcopy(pop)
        
        F = [if p["Cost"] >= 0
                1 / (1 + p["Cost"])
            else
                1 + abs(p["Cost"])
            end for p in pop]
        
        P = F / sum(F)
        
        for i in 1:nPop
            for k in 1:numbervar               
                j = RouletteWheelSelection(P)
                
                λ = alpha / (1 + P[j])
                
                newpop[i]["Position"][k] = pop[i]["Position"][k] +
                    λ * ((pop[j]["Position"][k] + elite[k]) / 2 - pop[i]["Position"][k])
                
                if rand() <= pMutation
                    newpop[i]["Position"][k] += sigma * randn()
                end
            end
            
            newpop[i]["Cost"] = costfcn(newpop[i]["Position"])
        end
        
        combined_pop = vcat(pop, newpop)
        combined_pop = sort!(combined_pop, by = x -> x["Cost"])
        pop = combined_pop[1:nPop]
        
        BestSol = pop[1]
        elite = BestSol["Position"]
        
        BestCost[it] = BestSol["Cost"]
    end
    return BestSol["Cost"], BestSol["Position"], BestCost
end
