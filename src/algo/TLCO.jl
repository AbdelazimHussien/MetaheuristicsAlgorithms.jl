"""
# References:

- Minh, Hoang-Le, Thanh Sang-To, Guy Theraulaz, Magd Abdel Wahab, and Thanh Cuong-Le.
  "Termite life cycle optimizer."
  Expert Systems with Applications 213 (2023): 119211.
"""
function TLCO(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)::OptimizationResult
    return TLCO(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter)
end

function TLCO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, cycle::Integer)
    dim = length(lb)
    if length(lb) == 1 && length(ub) == 1
        lb = fill(lb, dim)
        ub = fill(ub, dim)
    end

    trial_worker = zeros(npop)
    trial_soldier = zeros(npop)

    empt = Dict("pos" => [], "cost" => Inf)
    particle_termite = [deepcopy(empt) for _ = 1:npop]
    Int_Population = [deepcopy(empt) for _ = 1:npop]

    for i = 1:npop
        # Int_Population[i]["pos"] = lb .+ rand(dim) .* (ub - lb)
        Int_Population[i]["pos"] = initialization(1, dim, ub, lb)
        Int_Population[i]["cost"] = objfun(Int_Population[i]["pos"])
    end

    particle_soldier = deepcopy(Int_Population)
    particle_worker = deepcopy(Int_Population)
    particle_reproductive = deepcopy(Int_Population)

    costs = [ind["cost"] for ind in Int_Population]
    value, index = findmin(costs)

    gbest = Int_Population[index]

    best_TLCO = zeros(cycle)

    for k = 1:cycle
        beta = (0.5 / cycle) * k + 1.5

        sigma = 1 / (0.1 * cycle)
        lamda_soldier = 1 / (1 + exp(-sigma * (k - 0.5 * cycle)))
        lamda_worker = 1 - 1 / (1 + exp(-sigma * (k - 0.5 * cycle)))

        for i = 1:round(npop)
            if i <= round(0.7 * npop)
                particle_worker[i]["pos"] = particle_worker[i]["pos"] .+
                                            (-1 + 2 * rand()) * (rand(dim) .+ levy(1, dim, beta)) .*
                                            abs.(gbest["pos"] .- particle_worker[i]["pos"])

                particle_worker[i]["pos"] = min.(max.(particle_worker[i]["pos"], lb), ub)
                particle_worker[i]["cost"] = objfun(vec(particle_worker[i]["pos"]))

                if particle_worker[i]["cost"] < Int_Population[i]["cost"]
                    Int_Population[i] = deepcopy(particle_worker[i])

                    if Int_Population[i]["cost"] < gbest["cost"]
                        gbest = deepcopy(Int_Population[i])
                    end
                else
                    trial_worker[i] += 1
                end

                if trial_worker[i] / cycle > lamda_worker
                    trial_worker[i] = 0
                    A = setdiff(1:npop, i)
                    r1 = rand(1:npop)
                    A = setdiff(A, r1)
                    r2 = rand(1:npop)

                    if Int_Population[r2]["cost"] < Int_Population[r1]["cost"]
                        particle_reproductive[i]["pos"] = Int_Population[r1]["pos"] .+
                                                          rand(dim) .* (Int_Population[r2]["pos"] .- Int_Population[r1]["pos"])
                    else
                        particle_reproductive[i]["pos"] = Int_Population[r1]["pos"] .-
                                                          rand(dim) .* (Int_Population[r2]["pos"] .- Int_Population[r1]["pos"])
                    end

                    particle_reproductive[i]["pos"] = min.(max.(particle_reproductive[i]["pos"], lb), ub)
                    particle_reproductive[i]["cost"] = objfun(vec(particle_reproductive[i]["pos"]))

                    if particle_reproductive[i]["cost"] < Int_Population[i]["cost"]
                        Int_Population[i] = deepcopy(particle_reproductive[i])

                        if Int_Population[i]["cost"] < gbest["cost"]
                            gbest = deepcopy(Int_Population[i])
                        end
                    else
                        particle_reproductive[i]["pos"] = lb .+ rand(dim) .* (ub - lb)
                    end

                    particle_worker[i] = deepcopy(particle_reproductive[i])
                end
            end

            if i > round(0.7 * npop) && i <= npop
                particle_soldier[i]["pos"] = 2 * rand() * gbest["pos"] .+
                                             (-1 + 2 * rand()) .* abs.(particle_soldier[i]["pos"] .- levy(1, dim, beta) .* gbest["pos"])

                particle_soldier[i]["pos"] = min.(max.(particle_soldier[i]["pos"], lb), ub)

                particle_soldier[i]["cost"] = objfun(vec(particle_soldier[i]["pos"]))

                if particle_soldier[i]["cost"] < Int_Population[i]["cost"]
                    Int_Population[i] = deepcopy(particle_soldier[i])

                    if Int_Population[i]["cost"] < gbest["cost"]
                        gbest = deepcopy(Int_Population[i])
                    end
                else
                    trial_soldier[i] += 1
                end

                if trial_soldier[i] / cycle > lamda_soldier
                    trial_soldier[i] = 0
                    A = setdiff(1:npop, i)
                    r1 = rand(1:npop)
                    A = setdiff(A, r1)
                    r2 = rand(1:npop)

                    if Int_Population[r2]["cost"] < Int_Population[r1]["cost"]
                        particle_reproductive[i]["pos"] = Int_Population[r1]["pos"] .+
                                                          rand(dim) .* (Int_Population[r2]["pos"] .- Int_Population[r1]["pos"])
                    else
                        particle_reproductive[i]["pos"] = Int_Population[r1]["pos"] .-
                                                          rand(dim) .* (Int_Population[r2]["pos"] .- Int_Population[r1]["pos"])
                    end

                    particle_reproductive[i]["pos"] = min.(max.(particle_reproductive[i]["pos"], lb), ub)

                    particle_reproductive[i]["cost"] = objfun(vec(particle_reproductive[i]["pos"]))

                    if particle_reproductive[i]["cost"] < Int_Population[i]["cost"]
                        Int_Population[i] = deepcopy(particle_reproductive[i])

                        if Int_Population[i]["cost"] < gbest["cost"]
                            gbest = deepcopy(Int_Population[i])
                        end
                    else
                        particle_reproductive[i]["pos"] .= lb .+ rand(dim) .* (ub - lb)
                    end
                    particle_soldier[i] = deepcopy(particle_reproductive[i])
                end
            end
        end

        best_TLCO[k] = gbest["cost"]
    end

    # return gbest["cost"], gbest["pos"], best_TLCO
    return OptimizationResult(
        gbest["pos"],
        gbest["cost"],
        best_TLCO)
end

function TLCO(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000)::OptimizationResult
    return TLCO(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

# function levy_fun_TLCO(n::Int, m::Int, beta::Float64)
#     num = gamma(1 + beta) * sin(pi * beta / 2)
#     den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)

#     sigma_u = (num / den)^(1 / beta)

#     u = rand(Normal(0, sigma_u), n, m)
#     v = rand(Normal(0, 1), n, m)

#     z = u ./ abs.(v) .^ (1 / beta)
#     return z
# end