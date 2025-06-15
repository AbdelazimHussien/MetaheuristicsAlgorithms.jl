"""
MiarNaeimi, Farid, Gholamreza Azizyan, and Mohsen Rashki. 
"Horse herd optimization algorithm: A nature-inspired algorithm for high-dimensional optimization problems." 
Knowledge-Based Systems 213 (2021): 106711.
"""


function HorseOA(nHourse, MaxIt, VarMin, VarMax, nVar, CostFunction)
    VarSize = (1, nVar)  # Size of Decision Variables Matrix
    
    VelMax = 0.1 * (VarMax .- VarMin)
    VelMin = -VelMax

    # Initialization
    function create_empty_horse()
        return Dict(:Position => [], :Cost => [], :Velocity => [], :Best => Dict(:Position => [], :Cost => Inf))
    end

    Hourse = [create_empty_horse() for _ in 1:nHourse]
    GlobalBest = Dict(:Cost => Inf)

    CostPositionCounter = zeros(nHourse, 2 + nVar)

    for i in 1:nHourse
        Hourse[i][:Position] = VarMin .+ (VarMax .- VarMin) .* rand(nVar)
        
        Hourse[i][:Velocity] = zeros(VarSize)
        
        Hourse[i][:Cost] = CostFunction(vec(Hourse[i][:Position]'))
        
        
        Hourse[i][:Best][:Position] = Hourse[i][:Position]
        Hourse[i][:Best][:Cost] = Hourse[i][:Cost]
        
        if Hourse[i][:Best][:Cost] < GlobalBest[:Cost]
            GlobalBest = Hourse[i][:Best]
        end

        CostPositionCounter[i, 1] = i                           
        CostPositionCounter[i, 2] = Hourse[i][:Best][:Cost]      
        CostPositionCounter[i, 3:end] = Hourse[i][:Best][:Position]  
    end

    BestCost = zeros(MaxIt)
    nfe = zeros(MaxIt)

    # HOA Main Loop
    w = 0.8
    phiD = 0.02
    phiI = 0.02

    g_Alpha = 1.50  
    d_Alpha = 0.5  
    h_Alpha = 1.5  
    g_Beta = 1.50  
    h_Beta = 0.9  
    s_Beta = 0.20  
    d_Beta = 0.20  

    g_Gamma = 1.50  
    h_Gamma = 0.50  
    s_Gamma = 0.10  
    i_Gamma = 0.30  
    d_Gamma = 0.10  
    r_Gamma = 0.05  

    g_Delta = 1.50  
    r_Delta = 0.10  

    for it in 1:MaxIt
        sort!(CostPositionCounter, dims=1)
        MeanPosition = mean(CostPositionCounter[1:nHourse, 2])
        BadPosition = mean(CostPositionCounter[Int((1 - phiD) * nHourse):nHourse, 3])
        GoodPosition = mean(CostPositionCounter[1:Int(phiI * nHourse), 3])
        
        for i in 1:nHourse
            CC = findfirst(x -> x == i, CostPositionCounter[:, 1])  
            
            if CC <= 0.1 * nHourse
                Hourse[i][:Velocity] = h_Alpha * rand(VarSize) .* (GlobalBest[:Position] .- Hourse[i][:Position]) -
                                    d_Alpha * rand(VarSize) .* Hourse[i][:Position] +
                                    g_Alpha * (0.95 + 0.1 * rand()) * (Hourse[i][:Best][:Position] .- Hourse[i][:Position])
            elseif CC <= 0.3 * nHourse
                Hourse[i][:Velocity] = s_Beta * rand(VarSize) .* (MeanPosition .- Hourse[i][:Position]) -
                                    d_Beta * rand(VarSize) .* (BadPosition .- Hourse[i][:Position]) +
                                    h_Beta * rand(VarSize) .* (GlobalBest[:Position] .- Hourse[i][:Position]) +
                                    g_Beta * (0.95 + 0.1 * rand()) * (Hourse[i][:Best][:Position] .- Hourse[i][:Position])
            elseif CC <= 0.6 * nHourse
                Hourse[i][:Velocity] = s_Gamma * rand(VarSize) .* (MeanPosition .- Hourse[i][:Position]) +
                                    r_Gamma * rand(VarSize) .* Hourse[i][:Position] -
                                    d_Gamma * rand(VarSize) .* (BadPosition .- Hourse[i][:Position]) +
                                    h_Gamma * rand(VarSize) .* (GlobalBest[:Position] .- Hourse[i][:Position]) +
                                    i_Gamma * rand(VarSize) .* (GoodPosition .- Hourse[i][:Position]) +
                                    g_Gamma * (0.95 + 0.1 * rand()) * (Hourse[i][:Best][:Position] .- Hourse[i][:Position])
            else
                Hourse[i][:Velocity] = r_Delta * rand(VarSize) .* Hourse[i][:Position] +
                                    g_Delta * (0.95 + 0.1 * rand()) * (Hourse[i][:Best][:Position] .- Hourse[i][:Position])
            end
            
            Hourse[i][:Velocity] = clamp.(Hourse[i][:Velocity], VelMin, VelMax)
            
            Hourse[i][:Position] = Hourse[i][:Position] .+ Hourse[i][:Velocity]
            
            IsOutside = (Hourse[i][:Position] .< VarMin) .| (Hourse[i][:Position] .> VarMax)
            Hourse[i][:Velocity][IsOutside] .= -Hourse[i][:Velocity][IsOutside]
            
            Hourse[i][:Position] = clamp.(Hourse[i][:Position], VarMin, VarMax)
            
            Hourse[i][:Cost] = CostFunction(Hourse[i][:Position])
            
            if Hourse[i][:Cost] < Hourse[i][:Best][:Cost]
                Hourse[i][:Best][:Position] = Hourse[i][:Position]
                Hourse[i][:Best][:Cost] = Hourse[i][:Cost]
                
                if Hourse[i][:Best][:Cost] < GlobalBest[:Cost]
                    GlobalBest = Hourse[i][:Best]
                end
            end
        end
        
        BestCost[it] = GlobalBest[:Cost]
        nfe[it] = it
        
        # Update Parameters
        d_Alpha *= w
        g_Alpha *= w
        d_Beta *= w
        s_Beta *= w
        g_Beta *= w
        d_Gamma *= w
        s_Gamma *= w
        r_Gamma *= w
        i_Gamma *= w
        g_Gamma *= w
        r_Delta *= w
        g_Delta *= w
    end
    return GlobalBest[:Cost], GlobalBest[:Position], BestCost
end