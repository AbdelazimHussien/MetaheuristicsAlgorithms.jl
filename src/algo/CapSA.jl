"""
# References: 

- Braik, Malik, Alaa Sheta, and Heba Al-Hiary. "A novel meta-heuristic search algorithm for solving optimization problems: capuchin search algorithm." Neural computing and applications 33, no. 7 (2021): 2515-2547.

"""
function CapSA(noP, maxite, LB, UB, dim, fobj)

    cg_curve = zeros(maxite)
    
    CapPos = initialization(noP, dim, UB, LB)
    
    v = 0.1 .* CapPos  
    v0 = zeros(noP, dim)
    CapFit = zeros(noP)

    for i = 1:noP
        CapFit[i] = fobj(CapPos[i, :])
    end

    Fit = CapFit

    fitCapSA, index = findmin(CapFit)
    CapBestPos = copy(CapPos)  
    gFoodPos = CapPos[index, :]  

    bf = 0.70  
    cr = 11.0  
    g = 9.81

    a1 = 1.250
    a2 = 1.5

    beta = [2, 11, 2]
    wmax = 0.8
    wmin = 0.1

    for t = 1:maxite
        tau = beta[1] * exp(-beta[2] * t / maxite) ^ beta[3]
        w = wmax - (wmax - wmin) * (t / maxite)
        fol = ceil.(Int, (noP - 1) .* rand(noP))

        for i = 1:noP
            for j = 1:dim
                v[i, j] = w * v[i, j] + a1 * (CapBestPos[i, j] - CapPos[i, j]) * rand() + a2 * (gFoodPos[j] - CapPos[i, j]) * rand()
            end
        end

        for i = 1:noP
            r = rand()
            if i < noP / 2
                if rand() >= 0.1
                    if r <= 0.15
                        CapPos[i, :] = gFoodPos + bf * ((v[i, :].^2) * sin(2 * rand() * 1.5)) / g
                    elseif r > 0.15 && r <= 0.30
                        CapPos[i, :] = gFoodPos + cr * bf * ((v[i, :].^2) * sin(2 * rand() * 1.5)) / g
                    elseif r > 0.30 && r <= 0.9
                        CapPos[i, :] = CapPos[i, :] + v[i, :]
                    elseif r > 0.9 && r <= 0.95
                        CapPos[i, :] = gFoodPos .+ bf * sin(rand() * 1.5)
                    elseif r > 0.95
                        CapPos[i, :] = gFoodPos .+ bf * (v[i, :] - v0[i, :])
                    end
                else
                    CapPos[i, :] = tau * (LB .+ rand(dim) .* (UB .- LB))
                end
            elseif i >= noP / 2 && i <= noP
                eps = ((rand() + 2 * rand()) - (3 * rand())) / (1 + rand())
                Pos = gFoodPos .+ 2 * (CapBestPos[fol[i], :] - CapPos[i, :]) * eps .+ 2 * (CapPos[i, :] - CapBestPos[i, :]) * eps
                CapPos[i, :] = (Pos .+ CapPos[i - 1, :]) ./ 2
            end
        end

        v0 = copy(v)

        for i = 1:noP
            u = UB .- CapPos[i, :] .< 0
            l = LB .- CapPos[i, :] .> 0
            CapPos[i, :] = LB .* l .+ UB .* u .+ CapPos[i, :] .* .~xor.(u, l)
            CapFit[i] = fobj(CapPos[i, :])

            if CapFit[i] < Fit[i]
                CapBestPos[i, :] = CapPos[i, :]
                Fit[i] = CapFit[i]
            end
        end

        fmin, index = findmin(Fit)

        if fmin < fitCapSA
            gFoodPos = CapBestPos[index, :]
            fitCapSA = fmin
        end

        cg_curve[t] = fitCapSA
    end

    return fitCapSA, gFoodPos, cg_curve
end
