"""
Hu, Gang, Yuxuan Guo, Guo Wei, and Laith Abualigah. 
"Genghis Khan shark optimizer: a novel nature-inspired algorithm for engineering optimization." 
Advanced Engineering Informatics 58 (2023): 102210.
"""
function GKSO(n, Max_iter, lb, ub, dim, fhd)
    lb = ones(dim) .* lb  
    ub = ones(dim) .* ub  
    VarSize = (dim)       

    PopPos = zeros(n, dim)
    PopFit = zeros(n)
    for i in 1:n
        PopPos[i, :] = rand(dim) .* (ub - lb) + lb
        PopFit[i] = fhd(vec(PopPos[i, :]'))
    end

    Best_score = Inf
    Best_pos = []

    for i in 1:n
        if PopFit[i] <= Best_score
            Best_score = PopFit[i]
            Best_pos = PopPos[i, :]
        end
    end

    curve = zeros(Max_iter)
    h = [0.1]

    for it in 1:Max_iter
        h = append!(h, 1 - 2 * h[end]^4)
        p = 2 * (1 - (it / Max_iter)^(1 / 4)) + abs(h[end]) * ((it / Max_iter)^(1 / 4) - (it / Max_iter)^3)
        beta = 0.2 + (1.2 - 0.2) * (1 - (it / Max_iter)^3)^2
        alpha = abs(beta * sin((3 * π / 2 + sin(3 * π / 2 * beta))))

        newPopPos = zeros(n, dim)
        newPopFit = zeros(n)
        for i in 1:n
            newPopPos[i, :] = PopPos[i, :] + (lb + rand() * (ub - lb)) / it
            newPopPos[i, :] = clamp.(newPopPos[i, :], lb, ub)
            newPopFit[i] = fhd(vec(newPopPos[i, :]'))
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]  
                PopPos[i, :] = newPopPos[i, :]  
            end
        end

        GKS_Pos = zeros(n, dim)
        for i in 1:n
            s = (1.5 * PopFit[i]^rand())
            s = real(s)
            if i == 1
                newPopPos[i, :] = (Best_pos - PopPos[i, :]) * s
            else
                GKS_Pos[i, :] = (Best_pos - PopPos[i, :]) * s
                newPopPos[i, :] = (GKS_Pos[i, :] + newPopPos[i - 1, :]) / 2
            end
            if all(newPopPos[i, :] .>= lb) && all(newPopPos[i, :] .<= ub)
                newPopFit[i] = fhd(vec(newPopPos[i, :]'))
                if newPopFit[i] < PopFit[i]
                    PopPos[i, :] = newPopPos[i, :]
                    PopFit[i] = newPopFit[i]
                end
            end
        end

        for i in 1:n
            TF = (rand() > 0.5) * 2 - 1
            newPopPos[i, :] = Best_pos + rand(dim) .* (Best_pos - PopPos[i, :]) + TF * p^2 * (Best_pos - PopPos[i, :])
            newPopPos[i, :] = clamp.(newPopPos[i, :], lb, ub)
            newPopFit[i] = fhd(vec(newPopPos[i, :]'))
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        for i in 1:n
            A1 = rand(1:n, n)  
            r1, r2 = A1[1], A1[2]  
            k = rand(1:n)  
            if rand() < 0.5
                k = rand(1:n)
                f1, f2 = -1 + 2 * rand(), -1 + 2 * rand()
                ro = alpha * (2 * rand() - 1)
                Xk = rand(dim) .* (ub - lb) + lb

                L1 = rand() < 0.5
                u1 = L1 * 2 * rand() + (1 - L1)
                u2 = L1 * rand() + (1 - L1)
                u3 = L1 * rand() + (1 - L1)
                L2 = rand() < 0.5
                Xp = (1 - L2) * PopPos[k, :] + L2 * Xk
                popi1 = rand(VarSize) .* (ub - lb) + lb
                popi2 = rand(VarSize) .* (ub - lb) + lb

                A1 = rand(1:n, 2)  
                r1, r2 = A1[1], A1[2]
                k = rand(1:n)  

                
                if u1 < 0.5
                    newPopPos[i, :] = newPopPos[i, :] + f1 * (u1 * Best_pos - u2 * Xp) + f2 * ro * (u3 * (popi2 - popi1) + u2 * (PopPos[r1, :] - PopPos[r2, :])) / 2
                else
                    newPopPos[i, :] = Best_pos + f1 * (u1 * Best_pos - u2 * Xp) + f2 * ro * (u3 * (popi2 - popi1) + u2 * (PopPos[r1, :] - PopPos[r2, :])) / 2
                end
            end
            newPopPos[i, :] = clamp.(newPopPos[i, :], lb, ub)
            newPopFit[i] = fhd(vec(newPopPos[i, :]'))
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        for i in 1:n
            if PopFit[i] < Best_score
                Best_score = PopFit[i]
                Best_pos = PopPos[i, :]
            end
        end

        curve[it] = Best_score
    end

    return Best_score, Best_pos, curve
end