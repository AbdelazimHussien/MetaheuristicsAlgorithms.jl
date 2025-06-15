"""
Abdollahzadeh, Benyamin, Farhad Soleimanian Gharehchopogh, Nima Khodadadi, and Seyedali Mirjalili. 
"Mountain gazelle optimizer: a new nature-inspired metaheuristic algorithm for global optimization problems." 
Advances in Engineering Software 174 (2022): 103282.
"""


function Coefficient_Vector(dim, Iter, MaxIter)
    a2 = -1 + Iter * (-1 / MaxIter)
    u = randn(dim)  
    v = randn(dim)  

    
    cofi = zeros(Float64, 4, dim)  

    cofi[1, :] .= rand(dim)  
    cofi[2, :] .= (a2 + 1) .+ rand(dim)  
    cofi[3, :] .= a2 .* randn(dim)  
    cofi[4, :] .= u .* (v .^ 2) .* cos.(rand() * 2 .* u)  # Fill the fourth row

    return cofi
end

function Solution_Imp(X, BestX, lb, ub, N, cofi, M, A, D, i)
    NewX = zeros(Float64, 4, size(X, 2))  
    NewX[1, :] .= (ub .- lb) .* rand(size(X, 2)) .+ lb  
    NewX[2, :] .= BestX .- abs.((rand(1:2) .* M .- rand(1:2) .* X[i, :]) .* A) .* cofi[rand(1:4), :]  # Second row
    NewX[3, :] .= (M .+ cofi[rand(1:4), :]) .+ (rand(1:2) .* BestX .- rand(1:2) .* X[rand(1:N), :]) .* cofi[rand(1:4), :]  # Third row
    NewX[4, :] .= (X[i, :] .- D) .+ (rand(1:2) .* BestX .- rand(1:2) .* M) .* cofi[rand(1:4), :]  # Fourth row

    return NewX
end



function MountainGO(N, MaxIter, LB, UB, dim, fobj)
    lb = ones(dim) .* LB    # Lower Bound
    ub = ones(dim) .* UB    # Upper Bound

    X = initialization(N, dim, UB, LB)

    BestX = []
    BestFitness = Inf
    BestF = 0.0

    Sol_Cost = zeros(N) 
    for i in 1:N
        Sol_Cost[i] = fobj(X[i, :])  
        if Sol_Cost[i] <= BestFitness
            BestFitness = Sol_Cost[i]
            BestX = X[i, :]
        end
    end

    cnvg = zeros(MaxIter) 
    for Iter in 1:MaxIter
        for i in 1:N
            RandomSolution = rand(1:N, ceil(Int, N / 3))
            M = X[rand(ceil(Int, N / 3):N), :] * floor(rand()) + reshape(mean(X[RandomSolution, :], dims=1), 30) .* ceil(rand())

            cofi = Coefficient_Vector(dim, Iter, MaxIter)

            A = randn(dim) .* exp(2 - Iter * (2 / MaxIter))
            D = (abs.(X[i, :]) .+ abs.(BestX)) .* (2 * rand() .- 1)

            NewX = Solution_Imp(X, BestX, lb, ub, N, cofi, M, A, D, i)

            NewX .= max.(LB, min.(NewX, UB)) 
            Sol_CostNew = [fobj(row) for row in eachrow(NewX)]

            X = vcat(X, NewX)  
            Sol_Cost = vcat(Sol_Cost, Sol_CostNew)  

            _, idbest = findmin(Sol_Cost)
            BestX = X[idbest, :]
        end

        SortOrder = sortperm(Sol_Cost)
        Sol_Cost = Sol_Cost[SortOrder]

        X = X[SortOrder, :]
        BestFitness, idbest = findmin(Sol_Cost)
        BestX = X[idbest, :]
        X = X[1:N, :]
        Sol_Cost = Sol_Cost[1:N]
        cnvg[Iter] = BestFitness
        BestF = BestFitness
    end

    return BestF, BestX, cnvg
end
