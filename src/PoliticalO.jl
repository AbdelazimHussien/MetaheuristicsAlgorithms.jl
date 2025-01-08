"""
Askari, Qamar, Irfan Younas, and Mehreen Saeed. 
"Political Optimizer: A novel socio-inspired meta-heuristic for global optimization." 
Knowledge-based systems 195 (2020): 105709.
"""
    
function PoliticalO(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    parties = 8        
    lambda = 1.0   

    areas = Int(floor(sqrt(SearchAgents_no)))
    parties = Int(floor(sqrt(SearchAgents_no)))

    Leader_pos = zeros(1, dim)
    Leader_score = Inf  

    Positions = initialization(SearchAgents_no, dim, ub, lb)
    auxPositions = Positions
    prevPositions = Positions
    Convergence_curve = zeros(1, Max_iter)
    fitness = zeros(SearchAgents_no, 1)

    Positions, fitness, Leader_score, Leader_pos = Election(Positions, fitness, Leader_score, Leader_pos, ub, lb, fobj)

    auxFitness = fitness
    prevFitness = fitness
    aWinnerInd, aWinners, pLeaderInd, pLeaders = GovernmentFormation(Positions, fitness, areas, parties, dim)
    

    t = 0  
    while t < Max_iter
        prevFitness = auxFitness
        prevPositions = auxPositions
        auxFitness = fitness
        auxPositions = Positions

        ElectionCampaign!(Positions, prevPositions, fitness, prevFitness, pLeaders, aWinners, areas, parties, dim)
        PartySwitching!(Positions, fitness, parties, areas, lambda, t, Max_iter)
        Positions, fitness, Leader_score, Leader_pos = Election(Positions, fitness, Leader_score, Leader_pos, ub, lb, fobj)
        aWinnerInd, aWinners, pLeaderInd, pLeaders = GovernmentFormation(Positions, fitness, areas, parties, dim)
        Parliamentarism!(Positions, fitness, aWinners, aWinnerInd, areas, dim, fobj)
        
        t += 1
        Convergence_curve[t] = Leader_score
        # println(t, " ", Leader_score)
    end
    
    return Leader_score, Leader_pos, Convergence_curve
end

function Election(Positions, fitness, Leader_score, Leader_pos, ub, lb, fobj)
    ub = broadcast(identity, ub)  
    lb = broadcast(identity, lb)

    for i in axes(Positions, 1)
        Positions[i, :] = clamp.(Positions[i, :], lb, ub)

        fitness[i] = fobj(vec(Positions[i, :]'))

        if fitness[i] < Leader_score  
            Leader_score = fitness[i]
            Leader_pos = Positions[i, :]  
        end
    end

    return Positions, fitness, Leader_score, Leader_pos
end

function GovernmentFormation(Positions, fitness, areas, parties, dim)
    aWinnerInd = zeros(Int, areas)  
    aWinners = zeros(areas, dim)    

    for a in 1:areas
        aFitnessSlice = fitness[a:areas:end]
        _, aWinnerParty = findmin(aFitnessSlice)
        
        aWinnerInd[a] = (aWinnerParty - 1) * areas + a
        aWinners[a, :] = Positions[aWinnerInd[a], :]
    end

    pLeaderInd = zeros(Int, parties)  
    pLeaders = zeros(parties, dim)    

    for p in 1:parties
        pStIndex = (p - 1) * areas + 1
        pEndIndex = pStIndex + areas - 1
        pFitnessSlice = fitness[pStIndex:pEndIndex]
        
        _, leadIndex = findmin(pFitnessSlice)
        
        pLeaderInd[p] = (pStIndex - 1) + leadIndex  
        pLeaders[p, :] = Positions[pLeaderInd[p], :]
    end

    return aWinnerInd, aWinners, pLeaderInd, pLeaders
end

function ElectionCampaign!(Positions, prevPositions, fitness, prevFitness, pLeaders, aWinners, areas, parties, dim)
    for whichMethod in 1:2
        for a in 1:areas
            for p in 1:parties
                i = (p - 1) * areas + a  
                
                for j in 1:dim
                    center = if whichMethod == 1
                        pLeaders[p, j]  
                    elseif whichMethod == 2
                        aWinners[a, j]  
                    end

                    if prevFitness[i] >= fitness[i]  
                        if (prevPositions[i, j] <= Positions[i, j] <= center) || 
                           (prevPositions[i, j] >= Positions[i, j] >= center)
                            radius = center - Positions[i, j]
                            Positions[i, j] = center + rand() * radius
                        elseif (prevPositions[i, j] <= Positions[i, j] >= center && center >= prevPositions[i, j]) || 
                               (prevPositions[i, j] >= Positions[i, j] <= center && center <= prevPositions[i, j])
                            radius = abs(Positions[i, j] - center)
                            Positions[i, j] = center + (2 * rand() - 1) * radius
                        elseif (prevPositions[i, j] <= Positions[i, j] >= center && center <= prevPositions[i, j]) || 
                               (prevPositions[i, j] >= Positions[i, j] <= center && center >= prevPositions[i, j])
                            radius = abs(prevPositions[i, j] - center)
                            Positions[i, j] = center + (2 * rand() - 1) * radius
                        end
                    elseif prevFitness[i] < fitness[i]  
                        if (prevPositions[i, j] <= Positions[i, j] <= center) || 
                           (prevPositions[i, j] >= Positions[i, j] >= center)
                            radius = abs(Positions[i, j] - center)
                            Positions[i, j] = center + (2 * rand() - 1) * radius
                        elseif (prevPositions[i, j] <= Positions[i, j] >= center && center >= prevPositions[i, j]) || 
                               (prevPositions[i, j] >= Positions[i, j] <= center && center <= prevPositions[i, j])
                            radius = Positions[i, j] - prevPositions[i, j]
                            Positions[i, j] = prevPositions[i, j] + rand() * radius
                        elseif (prevPositions[i, j] <= Positions[i, j] >= center && center <= prevPositions[i, j]) || 
                               (prevPositions[i, j] >= Positions[i, j] <= center && center >= prevPositions[i, j])
                            center2 = prevPositions[i, j]
                            radius = abs(center - center2)
                            Positions[i, j] = center + (2 * rand() - 1) * radius
                        end
                    end
                end
            end
        end
    end
end

function PartySwitching!(Positions, fitness, parties, areas, lambda, t, Max_iter)
    psr = (1 - t * (1 / Max_iter)) * lambda

    for p in 1:parties
        for a in 1:areas
            fromPInd = (p - 1) * areas + a  

            if rand() < psr
                toParty = rand(1:parties)
                while toParty == p
                    toParty = rand(1:parties)
                end

                toPStInd = (toParty - 1) * areas + 1
                toPEndIndex = toPStInd + areas - 1
                _, toPLeastFit = findmax(fitness[toPStInd:toPEndIndex])
                toPInd = toPStInd + toPLeastFit - 1

                tempPosition = Positions[toPInd, :]
                Positions[toPInd, :] = Positions[fromPInd, :]
                Positions[fromPInd, :] = tempPosition

                tempFitness = fitness[toPInd]
                fitness[toPInd] = fitness[fromPInd]
                fitness[fromPInd] = tempFitness
            end
        end
    end
end

function Parliamentarism!(Positions, fitness, aWinners, aWinnerInd, areas, dim, fobj)
    for a in 1:areas
        newAWinner = copy(aWinners[a, :])  
        i = aWinnerInd[a]                 

        toa = rand(1:areas)
        while toa == a
            toa = rand(1:areas)
        end
        toAWinner = aWinners[toa, :]      

        for j in 1:dim
            distance = abs(toAWinner[j] - newAWinner[j])
            newAWinner[j] = toAWinner[j] + (2 * rand() - 1) * distance
        end

        newAWFitness = fobj(vec(newAWinner'))

        if newAWFitness < fitness[i]
            Positions[i, :] = newAWinner
            fitness[i] = newAWFitness
            aWinners[a, :] = newAWinner
        end
    end
end