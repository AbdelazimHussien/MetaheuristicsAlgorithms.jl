
function Exploration(Sol, lb, ub, dim, npop, objfun)
    sorted_indices = sortperm([s[:Cost] for s in Sol])
    Sol = [Sol[i] for i in sorted_indices]

    pCR = 0.20
    PCR = 1 - pCR
    p = PCR / npop

    NewSol = [Dict(:X => zeros(dim), :Cost => Inf) for _ = 1:npop]

    for i = 1:npop
        x = Sol[i][:X]
        A = randperm(npop)
        A = A[A.!=i]
        a, b, c, d, e, f = A[1:6]

        G = 2 * rand() - 1
        if rand() < 0.5
            y = rand(dim) .* (ub - lb) .+ lb
        else
            y = Sol[a][:X] + G * (Sol[a][:X] - Sol[b][:X]) +
                G * (((Sol[a][:X] - Sol[b][:X]) - (Sol[c][:X] - Sol[d][:X])) +
                     ((Sol[c][:X] - Sol[d][:X]) - (Sol[e][:X] - Sol[f][:X])))
        end

        y = clamp.(y, lb, ub)

        z = zeros(dim)
        j0 = rand(1:dim)
        for j = 1:dim
            if j == j0 || rand() <= pCR
                z[j] = y[j]
            else
                z[j] = x[j]
            end
        end

        NewSol[i][:X] = z
        NewSol[i][:Cost] = objfun(NewSol[i][:X])

        if NewSol[i][:Cost] < Sol[i][:Cost]
            Sol[i] = NewSol[i]
        else
            pCR += p
        end
    end

    return Sol
end

function Exploitation(Sol, lb, ub, dim, npop, Best, max_iter, Iter, objfun)
    Q = 0.67
    Beta = 2

    NewSol = [Dict(:X => zeros(dim), :Cost => Inf) for _ = 1:npop]

    mbest = mean([s[:X] for s in Sol]) / npop

    for i = 1:npop
        beta1 = 2 * rand()
        beta2 = randn(dim)
        w = randn(dim)
        v = randn(dim)
        F1 = randn(dim) .* exp(2 - Iter * (2 / max_iter))
        F2 = w .* v .^ 2 .* cos.(2 * rand() * w)
        R_1 = 2 * rand() - 1
        S1 = (2 * rand(dim) .- 1) .+ randn(dim)
        S2 = F1 .* R_1 .* Sol[i][:X] .+ F2 .* (1 - R_1) .* Best[:X]

        VEC = S2 ./ S1

        if rand() <= 0.5
            Xattack = VEC
            if rand() > Q
                NewSol[i][:X] = Best[:X] .+ beta1 .* exp.(beta2) .* (Sol[rand(1:npop)][:X] - Sol[i][:X])  # Eq 32
            else
                NewSol[i][:X] = beta1 .* Xattack .- Best[:X]
            end
        else
            r1 = round(Int, 1 + (npop - 1) * rand())
            NewSol[i][:X] = (mbest .* Sol[r1][:X] .- ((-1)^(rand(Bool))) .* Sol[i][:X]) ./ (1 + (Beta * rand()))  # Eq 32
        end

        NewSol[i][:X] = clamp.(NewSol[i][:X], lb, ub)
        NewSol[i][:Cost] = objfun(NewSol[i][:X])

        if NewSol[i][:Cost] < Sol[i][:Cost]
            Sol[i] = NewSol[i]
        end
    end

    return Sol
end

"""
# References:

- Abdollahzadeh, Benyamin, Nima Khodadadi, Saeid Barshandeh, Pavel Trojovský, Farhad Soleimanian Gharehchopogh, El-Sayed M. El-kenawy, Laith Abualigah, and Seyedali Mirjalili.
  "Puma optimizer (PO): A novel metaheuristic optimization algorithm and its application in machine learning."
  Cluster Computing (2024): 1-49.
"""
function PumaO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return PumaO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function PumaO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    UnSelected = ones(Float64, 2)
    F3_Explore = 0.0
    F3_Exploit = 0.0
    Seq_Time_Explore = ones(Float64, 3)
    Seq_Time_Exploit = ones(Float64, 3)
    Seq_Cost_Explore = ones(Float64, 3)
    Seq_Cost_Exploit = ones(Float64, 3)
    Score_Explore = 0.0
    Score_Exploit = 0.0
    PF = [0.5, 0.5, 0.3]
    PF_F3 = Float64[]
    Mega_Explor = 0.99
    Mega_Exploit = 0.99

    Convergence = zeros(max_iter)

    Sol = Vector{Any}(undef, npop)
    for i = 1:npop
        Sol[i] = Dict(:X => initialization(1, dim, ub, lb), :Cost => objfun(rand(lb:ub, dim)))
    end

    ind = argmin([s[:Cost] for s in Sol])
    Best = Sol[ind]
    Flag_Change = 1
    Initial_Best = Best

    Costs_Explor = zeros(Float64, 1, 3)
    Costs_Exploit = zeros(Float64, 1, 3)

    for Iter = 1:3
        Sol_Explor = Exploration(Sol, lb, ub, dim, npop, objfun)
        Costs_Explor[1, Iter] = minimum([s[:Cost] for s in Sol_Explor])

        Sol_Exploit = Exploitation(Sol, lb, ub, dim, npop, Best, max_iter, Iter, objfun)
        Costs_Exploit[1, Iter] = minimum([s[:Cost] for s in Sol_Exploit])

        Sol = vcat(Sol, Sol_Explor, Sol_Exploit)
        Sol = sort!(Sol, by=x -> x[:Cost])
        Best = Sol[1]

        Convergence[Iter] = Best[:Cost]
    end

    Seq_Cost_Explore[1] = abs(Initial_Best[:Cost] - Costs_Explor[1])
    Seq_Cost_Exploit[1] = abs(Initial_Best[:Cost] - Costs_Exploit[1])
    Seq_Cost_Explore[2] = abs(Costs_Explor[2] - Costs_Explor[1])
    Seq_Cost_Exploit[2] = abs(Costs_Exploit[2] - Costs_Exploit[1])
    Seq_Cost_Explore[3] = abs(Costs_Explor[3] - Costs_Explor[2])
    Seq_Cost_Exploit[3] = abs(Costs_Exploit[3] - Costs_Exploit[2])

    for i = 1:3
        if Seq_Cost_Explore[i] != 0
            push!(PF_F3, Seq_Cost_Explore[i])
        end
        if Seq_Cost_Exploit[i] != 0
            push!(PF_F3, Seq_Cost_Exploit[i])
        end
    end

    F1_Explor = PF[1] * (Seq_Cost_Explore[1] / Seq_Time_Explore[1])

    F1_Exploit = PF[1] * (Seq_Cost_Exploit[1] / Seq_Time_Exploit[1])

    F2_Explor = PF[2] * ((Seq_Cost_Explore[1] + Seq_Cost_Explore[2] + Seq_Cost_Explore[3]) /
                         (Seq_Time_Explore[1] + Seq_Time_Explore[2] + Seq_Time_Explore[3]))
    F2_Exploit = PF[2] * ((Seq_Cost_Exploit[1] + Seq_Cost_Exploit[2] + Seq_Cost_Exploit[3]) /
                          (Seq_Time_Explore[1] + Seq_Time_Explore[2] + Seq_Time_Explore[3]))
    Score_Explore = (PF[1] * F1_Explor) + (PF[2] * F2_Explor)
    Score_Exploit = (PF[1] * F1_Exploit) + (PF[2] * F2_Exploit)

    for Iter = 4:max_iter
        if Score_Explore > Score_Exploit
            SelectFlag = 1
            Sol = Exploration(Sol, lb, ub, dim, npop, objfun)
            Count_select = UnSelected
            UnSelected[2] += 1
            UnSelected[1] = 1
            F3_Explore = PF[3]
            F3_Exploit += PF[3]
            TBind = argmin([s[:Cost] for s in Sol])
            TBest = Sol[TBind]
            Seq_Cost_Explore[3] = Seq_Cost_Explore[2]
            Seq_Cost_Explore[2] = Seq_Cost_Explore[1]
            Seq_Cost_Explore[1] = abs(Best[:Cost] - TBest[:Cost])
            if Seq_Cost_Explore[1] != 0
                push!(PF_F3, Seq_Cost_Explore[1])
            end
            if TBest[:Cost] < Best[:Cost]
                Best = TBest
            end
        else
            SelectFlag = 2
            Sol = Exploitation(Sol, lb, ub, dim, npop, Best, max_iter, Iter, objfun)
            Count_select = UnSelected
            UnSelected[1] += 1
            UnSelected[2] = 1
            F3_Explore += PF[3]
            F3_Exploit = PF[3]

            TBind = argmin([s[:Cost] for s in Sol])
            TBest = Sol[TBind]
            Seq_Cost_Exploit[3] = Seq_Cost_Exploit[2]
            Seq_Cost_Exploit[2] = Seq_Cost_Exploit[1]
            Seq_Cost_Exploit[1] = abs(Best[:Cost] - TBest[:Cost])
            if Seq_Cost_Exploit[1] != 0
                push!(PF_F3, Seq_Cost_Exploit[1])
            end

            if TBest[:Cost] < Best[:Cost]
                Best = TBest
            end
        end

        if Flag_Change != SelectFlag
            Flag_Change = SelectFlag
            Seq_Time_Explore[3] = Seq_Time_Explore[2]
            Seq_Time_Explore[2] = Seq_Time_Explore[1]
            Seq_Time_Explore[1] = Count_select[1]
            Seq_Time_Exploit[3] = Seq_Time_Exploit[2]
            Seq_Time_Exploit[2] = Seq_Time_Exploit[1]
            Seq_Time_Exploit[1] = Count_select[2]
        end

        F1_Explor = PF[1] * (Seq_Cost_Explore[1] / Seq_Time_Explore[1])
        F1_Exploit = PF[1] * (Seq_Cost_Exploit[1] / Seq_Time_Exploit[1])
        F2_Explor = PF[2] * ((Seq_Cost_Explore[1] + Seq_Cost_Explore[2] + Seq_Cost_Explore[3]) /
                             (Seq_Time_Explore[1] + Seq_Time_Explore[2] + Seq_Time_Explore[3]))
        F2_Exploit = PF[2] * ((Seq_Cost_Exploit[1] + Seq_Cost_Exploit[2] + Seq_Cost_Exploit[3]) /
                              (Seq_Time_Exploit[1] + Seq_Time_Exploit[2] + Seq_Time_Exploit[3]))

        Score_Explore = (PF[1] * F1_Explor) + (PF[2] * F2_Explor)
        Score_Exploit = (PF[1] * F1_Exploit) + (PF[2] * F2_Exploit)
        # println("Iteration: $Iter Best Cost = $(Best[:Cost])")
        Convergence[Iter] = Best[:Cost]
    end

    # return Best[:Cost], Best[:X], Convergence
    return OptimizationResult(
        Best[:X],
        Best[:Cost],
        Convergence)
end

function PumaO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return PumaO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end