"""
# References:

- Wang, Liying, Qingjiao Cao, Zhenxing Zhang, Seyedali Mirjalili, and Weiguo Zhao. "Artificial rabbits optimization: A new bio-inspired meta-heuristic algorithm for solving engineering optimization problems." Engineering Applications of Artificial Intelligence 114 (2022): 105082.

"""
function ARO(nPop, MaxIt, Low, Up, Dim, F_index)#(F_index, MaxIt, nPop)

    PopPos = [rand(Dim) .* (Up - Low) .+ Low for _ in 1:nPop]
    PopFit = [F_index(PopPos[i]) for i in 1:nPop]

    BestF = Inf
    BestX = zeros(Dim)

    # Find initial best solution
    for i in 1:nPop
        if PopFit[i] <= BestF
            BestF = PopFit[i]
            BestX = PopPos[i]
        end
    end

    HisBestF = zeros(MaxIt)

    # Main loop
    for It in 1:MaxIt
        Direct1 = zeros(nPop, Dim)
        Direct2 = zeros(nPop, Dim)
        theta = 2 * (1 - It / MaxIt)

        for i in 1:nPop
            L = (exp(1) - exp(((It - 1) / MaxIt)^2)) * sin(2 * π * rand())
            rd = ceil(Int, rand() * Dim)
            Direct1[i, randperm(Dim)[1:rd]] .= 1
            c = Direct1[i, :]
            R = L .* c

            A = 2 * log(1 / rand()) * theta

            if A > 1
                K = [1:i-1; i+1:nPop]
                RandInd = K[rand(1:end)]
                newPopPos = PopPos[RandInd] .+ R .* (PopPos[i] .- PopPos[RandInd]) .+
                            round(0.5 * (0.05 + rand())) * randn(Dim)
            else
                Direct2[i, ceil(Int, rand() * Dim)] = 1
                gr = Direct2[i, :]
                H = ((MaxIt - It + 1) / MaxIt) * randn()
                b = PopPos[i] .+ H * gr .* PopPos[i]
                newPopPos = PopPos[i] .+ R .* (rand() .* b .- PopPos[i])
            end

            # Apply boundary constraints
            newPopPos = max.(min.(newPopPos, Up), Low)

            # Evaluate the new solution
            newPopFit = F_index(newPopPos)
            if newPopFit < PopFit[i]
                PopFit[i] = newPopFit
                PopPos[i] = newPopPos
            end
        end

        # Update the best solution
        for i in 1:nPop
            if PopFit[i] < BestF
                BestF = PopFit[i]
                BestX = PopPos[i]
            end
        end

        # Record the best fitness value
        HisBestF[It] = BestF
    end

    return BestF, BestX, HisBestF
end