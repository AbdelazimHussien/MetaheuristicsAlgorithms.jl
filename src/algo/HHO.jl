"""
Heidari, Ali Asghar, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, and Huiling Chen. 
"Harris hawks optimization: Algorithm and applications." 
Future generation computer systems 97 (2019): 849-872.
"""


function HHO(
    N::Int,
    T::Int,
    lb::Union{Int,AbstractVector},
    ub::Union{Int,AbstractVector},
    dim::Int,
    fobj::Function,
)
    Rabbit_Location = zeros(dim)
    Rabbit_Energy = Inf

    X = initialization(N, dim, ub, lb)

    CNVG = zeros(T)

    t = 0  # Loop counter

    while t < T
        for i = 1:N
            FU = X[i, :] .> ub
            FL = X[i, :] .< lb
            X[i, :] = (X[i, :] .* .~(FU .+ FL)) .+ ub .* FU .+ lb .* FL

            fitness = fobj(X[i, :])

            if fitness < Rabbit_Energy
                Rabbit_Energy = fitness
                Rabbit_Location = copy(X[i, :])
            end
        end

        E1 = 2 * (1 - (t / T))

        for i = 1:N
            E0 = 2 * rand() - 1
            Escaping_Energy = E1 * E0

            if abs(Escaping_Energy) >= 1
                q = rand()
                rand_Hawk_index = rand(1:N)
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5
                    X[i, :] = X_rand - rand() * abs.(X_rand - 2 * rand() * X[i, :])
                else
                    X[i, :] =
                        (Rabbit_Location .- vec(mean(X, dims=1))) .-
                        rand() * ((ub - lb) * rand() + lb)
                end

            elseif abs(Escaping_Energy) < 1
                r = rand()

                if r >= 0.5 && abs(Escaping_Energy) < 0.5
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs.(Rabbit_Location - X[i, :])
                elseif r >= 0.5 && abs(Escaping_Energy) >= 0.5
                    Jump_strength = 2 * (1 - rand())
                    X[i, :] =
                        (Rabbit_Location - X[i, :]) -
                        Escaping_Energy * abs.(Jump_strength * Rabbit_Location - X[i, :])
                elseif r < 0.5 && abs(Escaping_Energy) >= 0.5
                    Jump_strength = 2 * (1 - rand())
                    X1 =
                        Rabbit_Location -
                        Escaping_Energy * abs.(Jump_strength * Rabbit_Location - X[i, :])

                    if fobj(X1) < fobj(X[i, :])
                        X[i, :] = X1
                    else
                        X2 =
                            Rabbit_Location -
                            Escaping_Energy * abs.(Jump_strength * Rabbit_Location - X[i, :]) .+
                            rand(dim) .* Levy(dim)
                        if fobj(X2) < fobj(X[i, :])
                            X[i, :] = X2
                        end
                    end
                elseif r < 0.5 && abs(Escaping_Energy) < 0.5
                    Jump_strength = 2 * (1 - rand())
                    X1 =
                        Rabbit_Location .-
                        Escaping_Energy *
                        abs.(Jump_strength * Rabbit_Location .- vec(mean(X, dims=1)))

                    if fobj(X1) < fobj(X[i, :])
                        X[i, :] = X1
                    else
                        X2 =
                            Rabbit_Location -
                            Escaping_Energy *
                            abs.(Jump_strength * Rabbit_Location - vec(mean(X, dims=1))) .+
                            rand(dim) .* Levy(dim)
                        if fobj(X2) < fobj(X[i, :])
                            X[i, :] = X2
                        end
                    end
                end
            end
        end

        t += 1
        CNVG[t] = Rabbit_Energy
    end
    return Rabbit_Energy, Rabbit_Location, CNVG
end