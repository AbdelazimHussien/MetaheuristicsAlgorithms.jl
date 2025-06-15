"""
# References: 

- Wang, Xiaopeng, Václav Snášel, Seyedali Mirjalili, Jeng-Shyang Pan, Lingping Kong, and Hisham A. Shehadeh. "Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic algorithm for engineering optimization." Knowledge-Based Systems 295 (2024): 111737.

"""
function APO(pop_size, iter_max, Xmin, Xmax, dim, fhd)#(fhd, dim, pop_size, iter_max, Xmin, Xmax, Fid::Int, runid::Int)
    # Random seeds
    stm = sum(100 .* [year(now()), month(now()), day(now()), hour(now()), minute(now()), second(now())])
    Random.seed!(stm)

    # Global best
    targetbest = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]
    f_out_convergence = zeros(iter_max)

    ps = pop_size  # Protozoa size
    np = 1         # Neighbor pairs
    pf_max = 0.1   # Proportion fraction maximum

    # Set points to plot convergence curve
    for i in 1:51 # 51 points to plot
        iteration = (i == 1) ? 1 : div(iter_max, 50) * (i - 1)
    end

    # Initialize protozoa
    protozoa = zeros(Float64, ps, dim)    # Protozoa
    newprotozoa = zeros(Float64, ps, dim) # New protozoa
    epn = zeros(Float64, np, dim)         # Effect of paired neighbors
    protozoa_Fit = zeros(Float64, pop_size)

    # Initialization
    for i in 1:ps
        protozoa[i, :] = Xmin .+ rand(dim) .* (Xmax .- Xmin)
        protozoa_Fit[i] = fhd(vec(protozoa[i, :]'))
    end


    # Find the bestProtozoa and bestFit
    bestval, bestid = findmin(protozoa_Fit)
    bestProtozoa = protozoa[bestid, :]  # Best Protozoa
    bestFit = bestval                     # Best Fit
    f_out_convergence[1] = bestFit

    # Main loop  
    for iter in 2:iter_max
        index = sortperm(protozoa_Fit)
        protozoa_Fit = protozoa_Fit[index]
        protozoa .= protozoa[index, :]  # Reorder protozoa
        pf = pf_max * rand()              # Proportion fraction
        ri = randperm(ps)[1:ceil(Int, ps * pf)]  # Random indices

        for i in 1:ps
            if i in ri  # Protozoa in dormancy or reproduction form  
                pdr = 1/2 * (1 + cos((1 - i / ps) * π)) # Probability of dormancy and reproduction
                if rand() < pdr  # Dormancy form
                    newprotozoa[i, :] .= Xmin .+ rand(dim) .* (Xmax .- Xmin)
                else  # Reproduction form
                    flag = [1, -1]  # Plus-minus
                    Flag = flag[rand(1:2)]
                    Mr = zeros(Float64, dim) # Mapping vector in reproduction
                    Mr[randperm(dim)[1:ceil(Int, rand() * dim)]] .= 1
                    newprotozoa[i, :] .= protozoa[i, :] + Flag * rand() * (Xmin .+ rand(dim) .* (Xmax .- Xmin)) .* Mr
                end
            else  # Protozoa in foraging form
                f = rand() * (1 + cos(iter / iter_max * π)) # Foraging factor
                Mf = zeros(Float64, dim)  # Mapping vector in foraging
                Mf[randperm(dim)[1:ceil(Int, dim * i / ps)]] .= 1

                pah = 1/2 * (1 + cos(iter / iter_max * π)) # Probability of autotroph and heterotroph 
                if rand() < pah  # Autotroph form            
                    j = rand(1:ps)  # Randomly selected protozoa
                    for k in 1:np  # Neighbor pairs  
                        if i == 1
                            km = i  # k-
                            kp = i + rand(1:(ps - i))
                        elseif i == ps
                            km = rand(1:(ps - 1))
                            kp = i
                        else
                            km = rand(1:(i - 1))
                            kp = i + rand(1:(ps - i))
                        end

                        # Weight factor in the autotroph forms
                        wa = exp(-abs(protozoa_Fit[km] / (protozoa_Fit[kp] + eps())))
                        epn[k, :] .= wa * (protozoa[km, :] .- protozoa[kp, :])             
                    end                         
                    newprotozoa[i, :] .= protozoa[i, :] + f * (protozoa[j, :] .- protozoa[i, :] .+ (1 / np) * vec(sum(epn, dims=1))) .* Mf
                else   # Heterotroph form   
                    for k in 1:np  # Neighbor pairs 
                        if i == 1
                            imk = i   
                            ipk = i + k  
                        elseif i == ps
                            imk = ps - k
                            ipk = i
                        else
                            imk = i - k
                            ipk = i + k
                        end
                        # Neighbor limit range in [1, ps]
                        imk = max(1, imk)
                        ipk = min(ps, ipk)

                        # Weight factor in the heterotroph form
                        wh = exp(-abs(protozoa_Fit[imk] / (protozoa_Fit[ipk] + eps())))
                        epn[k, :] .= wh * (protozoa[imk, :] .- protozoa[ipk, :])
                    end           
                    flag = [1, -1]  # Plus-minus
                    Flag = flag[rand(1:2)]             
                    Xnear = (1 + Flag * rand(1:dim) * (1 - iter / iter_max)) .* protozoa[i, :]
                    newprotozoa[i, :] .= protozoa[i, :] + f * (Xnear .- protozoa[i, :] .+ (1 / np) * vec(sum(epn, dims=1))) .* Mf         
                end
            end
        end

        # Enforce bounds
        newprotozoa .= ((newprotozoa .>= Xmin) .& (newprotozoa .<= Xmax)) .* newprotozoa .+ 
                        (newprotozoa .< Xmin) .* Xmin .+ 
                        (newprotozoa .> Xmax) .* Xmax 

        newprotozoa_Fit = fhd.(eachrow(newprotozoa))

        bin = protozoa_Fit .> newprotozoa_Fit
        protozoa[bin, :] .= newprotozoa[bin, :]
        protozoa_Fit[bin] .= newprotozoa_Fit[bin]    

        bestFit, bestid = findmin(protozoa_Fit)
        bestProtozoa = protozoa[bestid, :]

        
        f_out_convergence[iter] = bestFit
 
    end
    
    return bestFit, bestProtozoa, f_out_convergence
end
