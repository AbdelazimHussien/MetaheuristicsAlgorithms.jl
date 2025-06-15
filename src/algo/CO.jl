mutable struct Individual
    Position::Vector{Float64}
    Cost::Float64
end


"""
# References:

- Akbari, Mohammad Amin, Mohsen Zare, Rasoul Azizipanah-Abarghooee, Seyedali Mirjalili, and Mohamed Deriche. "The cheetah optimizer: A nature-inspired metaheuristic algorithm for large-scale optimization problems." Scientific reports 12, no. 1 (2022): 10953.

"""
function CO(n, MaxIt, lb, ub, D, fobj)
    empty_individual = Individual([], Inf)
    BestSol = Individual([], Inf)
    pop = [empty_individual for _ in 1:n]

    MaxEEs = n * MaxIt

    xd = zeros(D)


    if length(lb) == 1
        ub = ub .* ones(D)    
        lb = lb .* ones(D)    
    end

    for i in 1:n
        pop[i].Position = lb .+ rand(D) .* (ub .- lb)
        pop[i].Cost = fobj(pop[i].Position)
        if pop[i].Cost < BestSol.Cost
            BestSol = pop[i]  
        end
    end

    pop1 = copy(pop)        
    BestCost = Float64[]    
    X_best = deepcopy(BestSol)
    Globest = Float64[]     

    t = 0                    
    it = 1                   
    T = ceil(Int, D / 10) * 60  
    FEs = 0                  

    while FEs <= MaxEEs
        m = rand(2:n)        
        i0 = rand(1:n, m)  
        for k in 1:m         
            i = rand(1:n)
            i = i0[k]

            if k == length(i0) && k != 1
                a = i0[k - 1]  
            else
                a = i0[k + 1]  
            end

            X = pop[i].Position    
            X1 = pop[a].Position    
            Xb = BestSol.Position    
            Xbest = X_best.Position  

            kk = 0
            if i <= 2 && t > 2 && t > Int(ceil(0.2 * T + 1)) && abs(BestCost[t - 2] - BestCost[t - Int(ceil(0.2 * T + 1))]) <= 0.0001 * Globest[t - 1]
                X = X_best.Position
                kk = 0
            elseif i == 3
                X = BestSol.Position
                kk = -0.1 * rand() * t / T
            else
                kk = 0.25
            end

            if it % 100 == 0 || it == 1
                xd = randperm(D)
            end

            Z = copy(X)

            for j in xd  
                r_Hat = randn()  
                r1 = rand()
                alpha = (k == 1) ? 0.0001 * t / T * (ub[j] - lb[j]) : 0.0001 * t / T * abs(Xb[j] - X[j]) + 0.001 * round(rand() > 0.9)

                r = randn()
                r_Check = abs(r)^exp(r / 2) * sin(2 * π * r)  
                beta = X1[j] - X[j]  

                h0 = exp(2 - 2 * t / T)
                H = abs(2 * r1 * h0 - h0)

                r2 = rand()
                r3 = kk + rand()

                if r2 <= r3
                    r4 = 3 * rand()
                    if H > r4
                        Z[j] = X[j] + r_Hat^-1 * alpha  
                    else
                        Z[j] = Xbest[j] + r_Check * beta  
                    end
                else
                    Z[j] = X[j]  
                end
            end

            Z = clamp.(Z, lb, ub)

            NewSol = Individual(Z, fobj(Z))
            if NewSol.Cost < pop[i].Cost
                pop[i] = NewSol
                if pop[i].Cost < BestSol.Cost
                    BestSol = pop[i]
                end
            end
            FEs += 1
        end

        t += 1  

        if t > T && t - round(T) - 1 >= 1 && t > 2
            if abs(BestCost[t - 1] - BestCost[t - round(T) - 1]) <= abs(0.01 * BestCost[t - 1])
                best = X_best.Position
                j0 = rand(1:D, ceil(Int, D / 10 * rand()))
                best[j0] = lb[j0] .+ rand(length(j0)) .* (ub[j0] .- lb[j0])
                BestSol.Cost = fobj(best)
                BestSol.Position = best  
                FEs += 1

                i0 = rand(1:n, round(1 * n))
                pop[i0[n - m + 1:n]] = pop1[i0[1:m]]  

                pop[i] = X_best  
                t = 1  
            end
        end

        it += 1  
        
        if BestSol.Cost < X_best.Cost
            X_best = BestSol
        end
        push!(BestCost, BestSol.Cost)
        push!(Globest, X_best.Cost)

    end

    return X_best.Cost, X_best.Position, BestCost
end