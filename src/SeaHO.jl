"""
Özbay, Feyza Altunbey. 
"A modified seahorse optimization algorithm based on chaotic maps for solving global optimization and engineering problems." 
Engineering Science and Technology, an International Journal 41 (2023): 101408.
"""

function SeaHO(pop, Max_iter, LB, UB, Dim, fobj)
    Sea_horses = initialization(pop, Dim, UB, LB)
    Sea_horsesFitness = zeros(pop)
    fitness_history = zeros(pop, Max_iter)
    population_history = zeros(pop, Dim, Max_iter)
    Convergence_curve = zeros(Max_iter)
    Trajectories = zeros(pop, Max_iter)
    
    for i in 1:pop
        Sea_horsesFitness[i] = fobj(Sea_horses[i, :])
        fitness_history[i, 1] = Sea_horsesFitness[i]
        population_history[i, :, 1] = Sea_horses[i, :]
        Trajectories[:, 1] .= Sea_horses[:, 1]
    end

    sorted_indexes = sortperm(Sea_horsesFitness)
    TargetPosition = Sea_horses[sorted_indexes[1], :]
    TargetFitness = Sea_horsesFitness[sorted_indexes[1]]
    Convergence_curve[1] = TargetFitness
    
    t = 1
    u = 0.05
    v = 0.05
    l = 0.05
    
    while t < Max_iter + 1
        beta = randn(pop, Dim)
        Elite = repeat(TargetPosition', pop, 1)
        r1 = randn(1, pop)
        Step_length = levy(pop, Dim, 1.5)
        Sea_horses_new1 = similar(Sea_horses)
        Sea_horses_new2 = similar(Sea_horses)
        Si = zeros(pop ÷ 2, Dim)

        for i in 1:pop
            for j in 1:Dim
                if r1[i] > 0
                    r = rand()
                    theta = r * 2 * pi
                    row = u * exp(theta * v)
                    x = row * cos(theta)
                    y = row * sin(theta)
                    z = row * theta
                    Sea_horses_new1[i, j] = Sea_horses[i, j] + Step_length[i, j] * ((Elite[i, j] - Sea_horses[i, j]) * x * y * z + Elite[i, j])
                else
                    Sea_horses_new1[i, j] = Sea_horses[i, j] + rand() * l * beta[i, j] * (Sea_horses[i, j] - beta[i, j] * Elite[i, j])
                end
            end
        end
        
        Sea_horses_new1 = max.(min.(Sea_horses_new1, UB'), LB')
        
        r2 = rand(pop)
        for i in 1:pop
            for j in 1:Dim
                alpha = (1 - t / Max_iter)^(2 * t / Max_iter)
                if r2[i] >= 0.1
                    Sea_horses_new2[i, j] = alpha * (Elite[i, j] - rand() * Sea_horses_new1[i, j]) + (1 - alpha) * Elite[i, j]
                else
                    Sea_horses_new2[i, j] = (1 - alpha) * (Sea_horses_new1[i, j] - rand() * Elite[i, j]) + alpha * Sea_horses_new1[i, j]
                end
            end
        end
        
        Sea_horses_new2 = max.(min.(Sea_horses_new2, UB'), LB')
        
        Sea_horsesFitness1 = [fobj(Sea_horses_new2[i, :]) for i in 1:pop]
        sorted_indexes = sortperm(Sea_horsesFitness1)
        
        Sea_horses_father = Sea_horses_new2[sorted_indexes[1:pop ÷ 2], :]
        Sea_horses_mother = Sea_horses_new2[sorted_indexes[pop ÷ 2 + 1:end], :]
        for k in 1:pop ÷ 2
            r3 = rand()
            Si[k, :] = r3 * Sea_horses_father[k, :] + (1 - r3) * Sea_horses_mother[k, :]
        end
        
        Sea_horses_offspring = Si
        Sea_horses_offspring = max.(min.(Sea_horses_offspring, UB'), LB')
        Sea_horsesFitness2 = [fobj(Sea_horses_offspring[i, :]) for i in 1:pop ÷ 2]
        
        Sea_horsesFitness = vcat(Sea_horsesFitness1, Sea_horsesFitness2)
        Sea_horses_new = vcat(Sea_horses_new2, Sea_horses_offspring)
        
        sorted_indexes = sortperm(Sea_horsesFitness)
        Sea_horses = Sea_horses_new[sorted_indexes[1:pop], :]
        SortfitbestN = Sea_horsesFitness[sorted_indexes[1:pop]]
        fitness_history[:, t] = SortfitbestN
        population_history[:, :, t] = Sea_horses
        Trajectories[:, t] = Sea_horses[:, 1]
        
        if SortfitbestN[1] < TargetFitness
            TargetPosition = Sea_horses[1, :]
            TargetFitness = SortfitbestN[1]
        end
        
        Convergence_curve[t] = TargetFitness
        t += 1
    end
    
    return TargetFitness, TargetPosition, Convergence_curve, Trajectories, fitness_history, population_history
end