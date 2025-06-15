"""
Han, Muxuan, Zunfeng Du, Kum Fai Yuen, Haitao Zhu, Yancang Li, and Qiuyu Yuan. 
"Walrus optimizer: A novel nature-inspired metaheuristic algorithm." 
Expert Systems with Applications 239 (2024): 122413.
"""


function levyFlight(d::Int)
    beta = 3/2

    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1/beta)
    u = randn(d) * sigma  
    v = randn(d)  
    step = u ./ abs.(v).^(1/beta)  

    return step
end

function WO(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    Best_Pos = zeros(Float64, dim)
    Second_Pos = zeros(Float64, dim)
    Best_Score = Inf
    Second_Score = Inf  
    GBestX = repeat(Best_Pos, SearchAgents_no, 1)


    X = initialization(SearchAgents_no, dim, ub, lb)

    Convergence_curve = zeros(Float64, Max_iter)

    P = 0.4  
    F_number = round(Int, SearchAgents_no * P)  
    M_number = F_number  
    C_number = SearchAgents_no - F_number - M_number  
    
    t = 0  

    while t < Max_iter
        for i in axes(X, 1)
            Flag4ub = X[i, :] .> ub
            Flag4lb = X[i, :] .< lb
            X[i, :] .= (X[i, :] .* .! (Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb  

            fitness = fobj(X[i, :])  

            if fitness < Best_Score
                Best_Score = fitness
                Best_Pos .= X[i, :]  
            end
            
            if fitness > Best_Score && fitness < Second_Score
                Second_Score = fitness
                Second_Pos .= X[i, :]  
            end
        end
        
        Alpha = 1 - t / Max_iter
        Beta = 1 - 1 / (1 + exp((1 / 2 * Max_iter - t) / Max_iter * 10))
        A = 2 * Alpha  
        r1 = rand()
        R = 2 * r1 - 1
        Danger_signal = A * R
        r2 = rand()
        Satey_signal = r2
        
        if abs(Danger_signal) >= 1
            r3 = rand()
            Rs = size(X, 1)
            Migration_step = (Beta * r3^2) * (X[randperm(Rs), :] .- X[randperm(Rs), :])
            X .= X .+ Migration_step

        elseif abs(Danger_signal) < 1
            if Satey_signal >= 0.5
                for i in 1:M_number
                    xy = zeros(Float64, M_number, 1)
                    base = 7
                    xy[i, 1] = hal(i, base)
                    m1 = lb .+ xy[i, :] .* (ub .- lb)
                    X[i, :] .= m1
                end
                for j in M_number + 1:M_number + F_number
                    X[j, :] .= X[j, :] .+ Alpha .* (X[i, :] .- X[j, :]) .+ (1 .- Alpha) .* (GBestX[j, :] .- X[j, :])
                end
                for i in SearchAgents_no - C_number + 1:SearchAgents_no
                    P = rand()
                    o = GBestX[i, :] .+ X[i, :] .* levyFlight(dim)
                    X[i, :] .= P * (o .- X[i, :])
                end
            end
                
            if Satey_signal < 0.5 && abs(Danger_signal) >= 0.5
                for i in 1:SearchAgents_no
                    r4 = rand()
                    X[i, :] .= X[i, :] .* R .- abs.(GBestX[i, :] .- X[i, :]) .* r4^2
                end
            end
            
            if Satey_signal < 0.5 && abs(Danger_signal) < 0.5
                for i in axes(X, 1)
                    for j in axes(X, 2)
                        theta1 = rand()
                        a1 = Beta * rand() - Beta
                        b1 = tan(theta1 * π)
                        X1 = Best_Pos[j] - a1 * b1 * abs(Best_Pos[j] - X[i, j])
                        
                        theta2 = rand()
                        a2 = Beta * rand() - Beta
                        b2 = tan(theta2 * π)
                        X2 = Second_Pos[j] - a2 * b2 * abs(Second_Pos[j] - X[i, j])
                        
                        X[i, j] = (X1 + X2) / 2
                    end
                end
            end
        end
        t += 1
        Convergence_curve[t] = Best_Score
    end

    return Best_Score, Best_Pos, Convergence_curve
end

function hal(index, base)
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0
        result += f * mod(i, base)
        i = div(i, base)  
        f /= base
    end
    return result
end