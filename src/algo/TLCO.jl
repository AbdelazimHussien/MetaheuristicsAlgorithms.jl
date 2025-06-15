"""
Minh, Hoang-Le, Thanh Sang-To, Guy Theraulaz, Magd Abdel Wahab, and Thanh Cuong-Le. 
"Termite life cycle optimizer." 
Expert Systems with Applications 213 (2023): 119211.
"""


function TLCO(NP, cycle, Lower_B, Upper_B, nvar, fobj)
    lb = Lower_B * ones(nvar)  
    ub = Upper_B * ones(nvar)  
    trial_worker = zeros(NP)    
    trial_soldier = zeros(NP)   

    empt = Dict("pos" => [], "cost" => Inf)
    particle_termite = [deepcopy(empt) for _ in 1:NP]  
    Int_Population = [deepcopy(empt) for _ in 1:NP]    

    for i in 1:NP
        Int_Population[i]["pos"] = lb .+ rand(nvar) .* (ub - lb)
        Int_Population[i]["cost"] = fobj(Int_Population[i]["pos"])
    end

    particle_soldier = deepcopy(Int_Population)
    particle_worker = deepcopy(Int_Population)
    particle_reproductive = deepcopy(Int_Population)

    costs = [ind["cost"] for ind in Int_Population]
    value, index = findmin(costs)

    Gbest = Int_Population[index]

    best_TLCO = zeros(cycle)

    for k in 1:cycle   
        beta = (0.5 / cycle) * k + 1.5  

        sigma = 1 / (0.1 * cycle)       
        lamda_soldier = 1 / (1 + exp(-sigma * (k - 0.5 * cycle)))  
        lamda_worker = 1 - 1 / (1 + exp(-sigma * (k - 0.5 * cycle)))  
        
        for i in 1:round(NP)
            if i <= round(0.7 * NP)
                particle_worker[i]["pos"] = particle_worker[i]["pos"] .+ 
                (-1 + 2 * rand()) * (rand(nvar) .+ levy_fun_TLCO(1, nvar, beta)) .* 
                abs.(Gbest["pos"] .- particle_worker[i]["pos"])
                
                particle_worker[i]["pos"] = min.(max.(particle_worker[i]["pos"], lb), ub)
                particle_worker[i]["cost"] = fobj(vec(particle_worker[i]["pos"]))

                if particle_worker[i]["cost"] < Int_Population[i]["cost"]
                    Int_Population[i] = deepcopy(particle_worker[i])
                    
                    if Int_Population[i]["cost"] < Gbest["cost"]
                        Gbest = deepcopy(Int_Population[i])  
                    end
                else
                    trial_worker[i] += 1
                end

                if trial_worker[i] / cycle > lamda_worker
                    trial_worker[i] = 0
                    A = setdiff(1:NP, i)
                    r1 = rand(1:NP)
                    A = setdiff(A, r1)
                    r2 = rand(1:NP)

                    if Int_Population[r2]["cost"] < Int_Population[r1]["cost"]
                        particle_reproductive[i]["pos"] = Int_Population[r1]["pos"] .+ 
                        rand(nvar) .* (Int_Population[r2]["pos"] .- Int_Population[r1]["pos"])
                    else
                        particle_reproductive[i]["pos"] = Int_Population[r1]["pos"] .- 
                        rand(nvar) .* (Int_Population[r2]["pos"] .- Int_Population[r1]["pos"])
                    end  

                    particle_reproductive[i]["pos"] = min.(max.(particle_reproductive[i]["pos"], lb), ub)
                    particle_reproductive[i]["cost"] = fobj(vec(particle_reproductive[i]["pos"]))

                    if particle_reproductive[i]["cost"] < Int_Population[i]["cost"]
                        Int_Population[i] = deepcopy(particle_reproductive[i])

                        if Int_Population[i]["cost"] < Gbest["cost"]
                            Gbest = deepcopy(Int_Population[i])  
                        end
                    else
                        particle_reproductive[i]["pos"] = lb .+ rand(nvar) .* (ub - lb)
                    end

                    particle_worker[i] = deepcopy(particle_reproductive[i])
                end          
            end 
             
            if i > round(0.7 * NP) && i <= NP
                particle_soldier[i]["pos"] = 2 * rand() * Gbest["pos"] .+ 
                (-1 + 2 * rand()) .* abs.(particle_soldier[i]["pos"] .- levy_fun_TLCO(1, nvar, beta) .* Gbest["pos"])

                particle_soldier[i]["pos"] = min.(max.(particle_soldier[i]["pos"], lb), ub)

                particle_soldier[i]["cost"] = fobj(vec(particle_soldier[i]["pos"]))

                if particle_soldier[i]["cost"] < Int_Population[i]["cost"]
                    Int_Population[i] = deepcopy(particle_soldier[i])

                    if Int_Population[i]["cost"] < Gbest["cost"]
                        Gbest = deepcopy(Int_Population[i])
                    end
                else
                    trial_soldier[i] += 1
                end
                
                if trial_soldier[i] / cycle > lamda_soldier
                    trial_soldier[i] = 0
                    A = setdiff(1:NP, i)
                    r1 = rand(1:NP)
                    A = setdiff(A, r1)
                    r2 = rand(1:NP)

                    if Int_Population[r2]["cost"] < Int_Population[r1]["cost"]
                        particle_reproductive[i]["pos"] = Int_Population[r1]["pos"] .+ 
                        rand(nvar) .* (Int_Population[r2]["pos"] .- Int_Population[r1]["pos"])
                    else
                        particle_reproductive[i]["pos"] = Int_Population[r1]["pos"] .- 
                        rand(nvar) .* (Int_Population[r2]["pos"] .- Int_Population[r1]["pos"])
                    end

                    particle_reproductive[i]["pos"] = min.(max.(particle_reproductive[i]["pos"], lb), ub)


                    particle_reproductive[i]["cost"] = fobj(vec(particle_reproductive[i]["pos"]))

                    if particle_reproductive[i]["cost"] < Int_Population[i]["cost"]
                        Int_Population[i] = deepcopy(particle_reproductive[i])

                        if Int_Population[i]["cost"] < Gbest["cost"]
                            Gbest = deepcopy(Int_Population[i])
                        end
                    else
                        particle_reproductive[i]["pos"] .= lb .+ rand(nvar) .* (ub - lb)
                    end
                    particle_soldier[i] = deepcopy(particle_reproductive[i])
                end    
            end 
        end 

        best_TLCO[k] = Gbest["cost"]
    end 


    return Gbest["cost"], Gbest["pos"], best_TLCO

end

function levy_fun_TLCO(n::Int, m::Int, beta::Float64)
    num = gamma(1 + beta) * sin(pi * beta / 2)  
    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)  

    sigma_u = (num / den)^(1 / beta)  

    u = rand(Normal(0, sigma_u), n, m)
    v = rand(Normal(0, 1), n, m)

    z = u ./ abs.(v).^(1 / beta)
    return z
end