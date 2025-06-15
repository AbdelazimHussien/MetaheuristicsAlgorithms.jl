"""
# References: 

- Xue J, Shen B. Dung beetle optimizer: A new meta-heuristic algorithm for global optimization. 
    The Journal of Supercomputing. 2023 May;79(7):7305-36.
    
"""
function DBO(pop, M, c, d, dim, fobj)
    P_percent = 0.2
    pNum = round(Int, pop * P_percent)  
    
    lb = fill(c, dim)
    ub = fill(d, dim)
    
    x = [lb .+ (ub .- lb) .* rand(dim) for _ in 1:pop]
    fit = [fobj(x[i]) for i in 1:pop]
    pFit = copy(fit)
    pX = copy(x)
    XX = copy(pX)
    
    fMin, bestI = findmin(fit)
    bestX = copy(x[bestI])
    Convergence_curve = zeros(M)
    
    for t in 1:M
        fmax, B = findmax(fit)
        worse = x[B]
        r2 = rand()
        
        for i in 1:pNum
            if r2 < 0.9
                r1 = rand()
                a = rand() > 0.1 ? 1 : -1
                x[i] = pX[i] .+ 0.3 .* abs.(pX[i] .- worse) .+ a * 0.1 .* XX[i]
            else
                theta = rand([0, 90, 180]) * π / 180
                x[i] = pX[i] .+ tan(theta) .* abs.(pX[i] .- XX[i])
            end
            x[i] = clamp.(x[i], lb, ub)
            fit[i] = fobj(x[i])
        end
        
        fMMin, bestII = findmin(fit)
        bestXX = copy(x[bestII])
        R = 1 - t / M
        
        Xnew1 = clamp.(bestXX .* (1 - R), lb, ub)
        Xnew2 = clamp.(bestXX .* (1 + R), lb, ub)
        Xnew11 = clamp.(bestX .* (1 - R), lb, ub)
        Xnew22 = clamp.(bestX .* (1 + R), lb, ub)
        
        for i in pNum + 1:12
            x[i] = bestXX .+ rand(dim) .* (pX[i] .- Xnew1) .+ rand(dim) .* (pX[i] .- Xnew2)
            x[i] = clamp.(x[i], Xnew1, Xnew2)
            fit[i] = fobj(x[i])
        end
        
        for i in 13:19
            x[i] = pX[i] .+ randn() .* (pX[i] .- Xnew11) .+ rand(dim) .* (pX[i] .- Xnew22)
            x[i] = clamp.(x[i], lb, ub)
            fit[i] = fobj(x[i])
        end
        
        for j in 20:pop
            x[j] = bestX .+ randn(dim) .* ((abs.(pX[j] .- bestXX) .+ abs.(pX[j] .- bestX)) ./ 2)
            x[j] = clamp.(x[j], lb, ub)
            fit[j] = fobj(x[j])
        end
        
        XX = copy(pX)
        for i in 1:pop
            if fit[i] < pFit[i]
                pFit[i] = fit[i]
                pX[i] = copy(x[i])
            end
            if pFit[i] < fMin
                fMin = pFit[i]
                bestX = copy(pX[i])
            end
        end
        Convergence_curve[t] = fMin
    end
    
    return fMin, bestX, Convergence_curve
end