"""
Moosavi, Seyyed Hamid Samareh, and Vahid Khatibi Bardsiri. 
"Poor and rich optimization algorithm: A new human-based and multi populations algorithm." 
Engineering applications of artificial intelligence 86 (2019): 165-181.
"""


function PRO(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction)
    VarSize = (1, nVar)

    npoor = div(nPop, 2) 
    nrich = div(nPop, 2) 
    pmut = 0.06              


    pop = [Dict("Position" => rand(nVar) .* (VarMax - VarMin) .+ VarMin,
                "Cost" => Inf) for _ in 1:nPop]

    for i in 1:nPop
        pop[i]["Cost"] = CostFunction(pop[i]["Position"])
    end
    pop = sort(pop, by = x -> x["Cost"])
    BestSol = pop[1]


    pop1 = pop[1:nrich]  
    pop2 = pop[nrich + 1:end]  

    BestCost = zeros(MaxIt)
    allpop = zeros(MaxIt, nPop)

    for it in 1:MaxIt
        newpop1 = deepcopy(pop1)
        newpop2 = deepcopy(pop2)

        for i in 1:nrich
            newpop1[i]["Position"] += rand() .* (pop1[i]["Position"] - pop2[1]["Position"])

            if rand() < pmut
                newpop1[i]["Position"] += randn(nVar)
            end

            newpop1[i]["Cost"] = CostFunction(newpop1[i]["Position"])
            newpop1[i]["Position"] = clamp.(newpop1[i]["Position"], VarMin, VarMax)
        end

        meanr = zeros(30)  

        for d in 1:nrich
            meanr .+= pop1[d]["Position"]  
        end

        meanr ./= nrich  


        pattern = (pop1[1]["Position"] + meanr + pop1[end]["Position"]) / 3

        for j in 1:npoor
            newpop2[j]["Position"] += rand() .* (pattern - pop2[j]["Position"])

            if rand() < pmut
                newpop2[j]["Position"] += randn(nVar)
            end

            newpop2[j]["Cost"] = CostFunction(newpop2[j]["Position"])
            newpop2[j]["Position"] = clamp.(newpop2[j]["Position"], VarMin, VarMax)
        end

        pop = vcat(pop1, pop2, newpop1, newpop2)

        pop = sort(pop, by = x -> x["Cost"])[1:nPop]
        pop1 = pop[1:nrich]
        pop2 = pop[nrich + 1:end]

        BestSol = pop1[1]
        BestCost[it] = BestSol["Cost"]

    end

    return BestSol["Cost"], BestSol, BestCost[end]

end

