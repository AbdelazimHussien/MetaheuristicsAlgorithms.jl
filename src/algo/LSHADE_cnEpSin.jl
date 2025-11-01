"""
"""
function LSHADE_cnEpSin(objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
    return LSHADE_cnEpSin(objfun, fill(lb, dim), fill(ub, dim), npop, max_iter) 
end
# function LSHADE_cnEpSin()
# function LSHADE_cnEpSin(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
function LSHADE_cnEpSin(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    
    problem_size = length(lb)
    # problem_size = 50
    freq_inti = 0.5
    # max_nfes = 10000 * problem_size
    max_nfes = npop * max_iter

    Random.seed!(round(Int, time()))  # seed random

    val_2_reach = 1e-8
    max_region = 100.0
    min_region = -100.0
    # lu = [fill(-100.0, problem_size); fill(100.0, problem_size)]
    lu = [lb; ub]

    pb = 0.4
    ps = 0.5

    S_Ndim = problem_size
    S_Lband = fill(-100.0, problem_size)
    S_Uband = fill(100.0, problem_size)

    G_Max = problem_size == 10 ? 2163 : problem_size == 30 ? 2745 : problem_size == 50 ? 3022 : problem_size == 100 ? 3401 : 0

    # num_prbs = 30
    # runs = 51

    RecordFEsFactor = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4,
                      0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    progress = length(RecordFEsFactor)

    # allerrorvals = zeros(Float64, progress, runs, num_prbs)
    # result = zeros(num_prbs,5)

    println("Running LSHADE_cnEpSin on D= $problem_size")

    # for func in 1:num_prbs
        # optimum = func * 100.0
        optimum = 1 * 100.0
        # S_FuncNo = func

        # #println("\n-------------------------------------------------------")
        # #println("Function = $func, Dimension size = $problem_size")

        # Using Threads.@threads or distributed parallelism can replace parfor here,
        # but for simplicity, we run sequentially.
        # for run_id in 1:runs

        run_funcvals = Float64[]
        col = 1

        # parameter settings for L-SHADE
        p_best_rate = 0.11
        arc_rate = 1.4
        memory_size = 5
        pop_size = 18 * problem_size
        SEL = round(Int, ps * pop_size)

        max_pop_size = pop_size
        min_pop_size = 4

        nfes = 0
        # Initialize population
        popold = lu[1, :] .+ rand(pop_size, problem_size) .* (lu[2, :] .- lu[1, :])
        pop = copy(popold)

        # fitness = cec17_func(pop', func)'
        # fitness = objfun(pop')'
        # fitness = objfun(vec(pop'))
        fitness = [objfun(vec(pop[i, :])) for i in 1:pop_size]
        bsf_fit_var = 1e30
        bsf_index = 0
        bsf_solution = zeros(problem_size)
        his_best_fit = zeros(Float64, max_nfes)

        # Initial evaluation count and best solution
        for i in 1:pop_size
            nfes += 1
            if fitness[i] < bsf_fit_var
                bsf_fit_var = fitness[i]
                bsf_solution .= pop[i, :]
                bsf_index = i
            end
            if nfes > max_nfes
                break
            end
        end

        memory_sf = fill(0.5, memory_size)
        memory_cr = fill(0.5, memory_size)
        memory_freq = fill(freq_inti, memory_size)
        memory_pos = 1

        # archive_NP = Int(arc_rate * pop_size)
        archive_NP = round(Int, arc_rate * pop_size)
        archive_pop = zeros(0, problem_size)
        archive_funvalues = zeros(0)

        gg = 0
        igen = 1

        flag1 = false
        flag2 = false

        goodF1all = Int[]
        goodF2all = Int[]
        badF1all = Int[]
        badF2all = Int[]

        goodF1 = Float64[]
        goodF2 = Float64[]
        badF1 = Float64[]
        badF2 = Float64[]

        while nfes < max_nfes
            gg += 1
            pop = copy(popold)

            temp_fit, sorted_index = sort(fitness), sortperm(fitness)

            mem_rand_index = rand(1:memory_size, pop_size)
            mu_sf = memory_sf[mem_rand_index]
            mu_cr = memory_cr[mem_rand_index]
            mu_freq = memory_freq[mem_rand_index]

            # Generate crossover rate cr from normal distribution N(mu_cr, 0.1)
            cr = mu_cr .+ 0.1 .* randn(pop_size)
            cr[mu_cr .== -1] .= 0
            cr = clamp.(cr, 0, 1)

            # Generate scaling factor sf from mu_sf + 0.1 * tan(pi*(rand-0.5))
            sf = similar(cr)
            for i in 1:pop_size
                val = mu_sf[i] + 0.1 * tan(pi*(rand() - 0.5))
                while val <= 0
                    val = mu_sf[i] + 0.1 * tan(pi*(rand() - 0.5))
                end
                sf[i] = min(val, 1)
            end

            # Generate freq similarly
            freq = similar(sf)
            for i in 1:pop_size
                val = mu_freq[i] + 0.1 * tan(pi*(rand() - 0.5))
                while val <= 0
                    val = mu_freq[i] + 0.1 * tan(pi*(rand() - 0.5))
                end
                freq[i] = min(val, 1)
            end

            LP = 20
            flag1 = false
            flag2 = false
            if nfes <= max_nfes/2
                if gg <= LP
                    p1 = 0.5
                    p2 = 0.5
                    c = rand()
                    if c < p1
                        sf .= 0.5 .* (sin.(2 .* pi .* freq_inti .* gg .+ pi) .* ((G_Max - gg)/G_Max) .+ 1)
                        flag1 = true
                    else
                        sf .= 0.5 .* (sin.(2 .* pi .* freq .* gg) .* (gg/G_Max) .+ 1)
                        flag2 = true
                    end
                else
                    ns1_sum = sum(goodF1all[end-LP+1:end])
                    nf1_sum = sum(badF1all[end-LP+1:end])
                    sumS1 = (ns1_sum/(ns1_sum + nf1_sum)) + 0.01

                    ns2_sum = sum(goodF2all[end-LP+1:end])
                    nf2_sum = sum(badF2all[end-LP+1:end])
                    sumS2 = (ns2_sum/(ns2_sum + nf2_sum)) + 0.01

                    p1 = sumS1/(sumS1 + sumS2)
                    p2 = sumS2/(sumS1 + sumS2)

                    c = rand()
                    if c < p1
                        sf .= 0.5 .* (sin.(2 .* pi .* freq_inti .* gg .+ pi) .* ((G_Max - gg)/G_Max) .+ 1)
                        flag1 = true
                    else
                        sf .= 0.5 .* (sin.(2 .* pi .* freq .* gg) .* (gg/G_Max) .+ 1)
                        flag2 = true
                    end
                end
            else
                sf .= 0.5 .* (sin.(2 .* pi .* freq_inti .* gg .+ pi) .* ((G_Max - gg)/G_Max) .+ 1)
                flag1 = true
            end

            # p_best selection (top p_best_rate)
            p_num = max(round(Int, p_best_rate * pop_size), 2)
            pbest_indices = sorted_index[1:p_num]

            # Prepare offspring matrix
            offspring = similar(pop)

            for i in 1:pop_size
                r0 = i
                r1, r2 = gnR1R2(pop_size, pop_size + size(archive_pop,1), [r0])

                # r1 and r2 are vectors but only one index per i, so take first element
                r1_i = r1[1]
                r2_i = r2[1]

                # Select pbest individual randomly
                pbest_index = pbest_indices[rand(1:p_num)]

                # Mutation
                vi = pop[i, :] .+ sf[i] .* (pop[pbest_index, :] .- pop[i, :]) .+ sf[i] .* (pop[r1_i, :] .- (r2_i > pop_size ? archive_pop[r2_i - pop_size, :] : pop[r2_i, :]))

                # Bound constrain
                vi = boundConstraint([vi], [pop[i, :]], lu)[1, :]

                # Crossover
                jrand = rand(1:problem_size)
                ui = zeros(problem_size)
                for j in 1:problem_size
                    if rand() <= cr[i] || j == jrand
                        ui[j] = vi[j]
                    else
                        ui[j] = pop[i, j]
                    end
                end
                offspring[i, :] = ui
            end

            # Evaluate offspring
            # fitness_offspring = cec17_func(offspring', func)'
            # fitness_offspring = objfun(offspring')'
            # fitness_offspring = objfun(vec(offspring'))
            fitness_offspring = [objfun(vec(offspring[i, :])) for i in 1:pop_size]

            nfes += pop_size


            # Selection and archive update
            goodF1all_ = 0
            goodF2all_ = 0
            badF1all_ = 0
            badF2all_ = 0

            goodF1_ = Float64[]
            goodF2_ = Float64[]
            badF1_ = Float64[]
            badF2_ = Float64[]

            newpop = similar(pop)
            newfit = similar(fitness)

            for i in 1:pop_size
                if fitness_offspring[i] <= fitness[i]
                    newpop[i, :] = offspring[i, :]
                    newfit[i] = fitness_offspring[i]

                    if flag1
                        push!(goodF1_, sf[i])
                        goodF1all_ += 1
                    elseif flag2
                        push!(goodF2_, sf[i])
                        goodF2all_ += 1
                    end

                    # Add replaced parent to archive
                    if size(archive_pop, 1) < archive_NP
                        archive_pop = vcat(archive_pop, reshape(pop[i, :], 1, problem_size))
                        archive_funvalues = vcat(archive_funvalues, fitness[i])
                    else
                        # Replace randomly
                        idx = rand(1:archive_NP)
                        archive_pop[idx, :] = pop[i, :]
                        archive_funvalues[idx] = fitness[i]
                    end
                else
                    newpop[i, :] = pop[i, :]
                    newfit[i] = fitness[i]

                    if flag1
                        push!(badF1_, sf[i])
                        badF1all_ += 1
                    elseif flag2
                        push!(badF2_, sf[i])
                        badF2all_ += 1
                    end
                end
            end

            pop = newpop
            fitness = newfit

            # Update archive records
            goodF1all = vcat(goodF1all, goodF1all_)
            goodF2all = vcat(goodF2all, goodF2all_)
            badF1all = vcat(badF1all, badF1all_)
            badF2all = vcat(badF2all, badF2all_)

            goodF1 = vcat(goodF1, goodF1_)
            goodF2 = vcat(goodF2, goodF2_)
            badF1 = vcat(badF1, badF1_)
            badF2 = vcat(badF2, badF2_)

            # Update memory
            if !isempty(goodF1)
                memory_sf[memory_pos] = mean(goodF1)
                memory_cr[memory_pos] = mean(cr)
                memory_freq[memory_pos] = mean(freq)
            else
                memory_sf[memory_pos] = 0.5
                memory_cr[memory_pos] = 0.5
                memory_freq[memory_pos] = freq_inti
            end

            memory_pos += 1
            if memory_pos > memory_size
                memory_pos = 1
            end

            # Update best solution
            current_best_idx = argmin(fitness)
            current_best_fit = fitness[current_best_idx]
            if current_best_fit < bsf_fit_var
                bsf_fit_var = current_best_fit
                bsf_solution .= pop[current_best_idx, :]
            end

            # Record progress if needed
            for p in 1:progress
                if nfes >= RecordFEsFactor[p] * max_nfes && length(run_funcvals) < p
                    push!(run_funcvals, bsf_fit_var - optimum)
                end
            end

            # Population size reduction
            if nfes >= max_nfes / 2
                if pop_size > min_pop_size
                    reduction_ind_num = max(round(Int, pop_size * (nfes - max_nfes/2) / (max_nfes/2)), 0)
                    new_pop_size = max(pop_size - reduction_ind_num, min_pop_size)

                    if new_pop_size < pop_size
                        sorted_indices = sortperm(fitness)
                        pop = pop[sorted_indices[1:new_pop_size], :]
                        fitness = fitness[sorted_indices[1:new_pop_size]]
                        pop_size = new_pop_size
                    end
                end
            end
            # his_best_fit[nfes] = bsf_fit_var 
            push!(his_best_fit, bsf_fit_var)
            println("$nfes --> $bsf_fit_var")

        end # while nfes < max_nfes

        # Save error vals for this run
        # for p in 1:progress
        #     if length(run_funcvals) >= p
        #         allerrorvals[p, run_id, func] = run_funcvals[p]
        #     else
        #         allerrorvals[p, run_id, func] = NaN
        #     end
        # end

        # println("Run $run_id best fitness = $bsf_fit_var")
        println("Run  best fitness = $bsf_fit_var")
        

        # end # runs

        # Summary for function
        # result[func, 1] = mean(allerrorvals[end, :, func], dims=1)[1]
        # result[func, 2] = std(allerrorvals[end, :, func], dims=1)[1]
        # result[func, 3] = minimum(allerrorvals[end, :, func])
        # result[func, 4] = maximum(allerrorvals[end, :, func])
        # result[func, 5] = median(allerrorvals[end, :, func])

        # println("Function $func summary:")
        # println("Mean error: ", result[func, 1])
        # println("Std error: ", result[func, 2])
        # println("Min error: ", result[func, 3])
        # println("Max error: ", result[func, 4])
        # println("Median error: ", result[func, 5])

    # end # num_prbs

    println("bestF: ", typeof(bsf_fit_var))
    println("bestX: ", typeof(bsf_solution))
    println("his_best_fit: ", typeof(his_best_fit))

    return OptimizationResult(
        bsf_solution,
        bsf_fit_var,
        his_best_fit
    )
end

function LSHADE_cnEpSin(problem::OptimizationProblem, npop::Integer=30, max_iter::Integer=1000
)::OptimizationResult
    return LSHADE_cnEpSin(problem.objfun, problem.lb, problem.ub, npop, max_iter)
end

function gnR1R2(NP1::Int, NP2::Int, r0::Vector{Int})
    NP0 = length(r0)
    r1 = rand(1:NP1, NP0)

    for i in 1:99999999
        pos = findall(x -> x, r1 .== r0)
        if isempty(pos)
            break
        else
            r1[pos] .= rand(1:NP1, length(pos))
        end
        if i > 1000
            error("Cannot generate r1 in 1000 iterations")
        end
    end

    r2 = rand(1:NP2, NP0)

    for i in 1:99999999
        pos = findall(x -> x, (r2 .== r0) .| (r2 .== r1))
        if isempty(pos)
            break
        else
            r2[pos] .= rand(1:NP2, length(pos))
        end
        if i > 1000
            error("Cannot generate r2 in 1000 iterations")
        end
    end

    return r1, r2
end

function updateArchive(archive, pop, funvalue)
    # archive must be a dictionary with keys: :NP, :pop, :funvalues

    if archive[:NP] == 0
        return archive
    end

    if size(pop, 1) != size(funvalue, 1)
        error("check it")
    end

    # Step 1: Add new solutions to the archive
    popAll = vcat(archive[:pop], pop)
    funvalues = vcat(archive[:funvalues], funvalue)

    # Step 2: Remove duplicate rows in popAll and sync funvalues
    # _, IX = unique(popAll; dims=1, return_index=true)
    # popAll = popAll[sort(IX), :]

    popAll, IX = unique(popAll; dims=1, return_index=true)
    # popAll, IX = unique(popAll, dims=1, returninds=true)
    popAll = popAll[sortperm(IX), :]

    funvalues = funvalues[sort(IX), :]

    # Step 3: Truncate if needed
    if size(popAll, 1) <= archive[:NP]
        archive[:pop] = popAll
        archive[:funvalues] = funvalues
    else
        rndpos = randperm(size(popAll, 1))[1:archive[:NP]]
        archive[:pop] = popAll[rndpos, :]
        archive[:funvalues] = funvalues[rndpos, :]
    end

    return archive
end


function boundConstraint(vi, pop, lu::AbstractVector{<:Real})
    # Convert lu to 2×D matrix if it is a flat vector (concatenated lb and ub)
    D = length(lu) ÷ 2
    lu = reshape(lu, 2, D)

    # Convert pop and vi to matrices if they are vectors of vectors
    if isa(vi, Vector{Vector{Float64}})
        vi = hcat(vi...)'  # (D × NP) → (NP × D)
    end
    if isa(pop, Vector{Vector{Float64}})
        pop = hcat(pop...)'
    end

    NP, D = size(pop)
    xl = repeat(lu[1:1, :], NP, 1)
    xu = repeat(lu[2:2, :], NP, 1)

    pos_lower = vi .< xl
    vi[pos_lower] .= (pop[pos_lower] + xl[pos_lower]) ./ 2

    pos_upper = vi .> xu
    vi[pos_upper] .= (pop[pos_upper] + xu[pos_upper]) ./ 2

    return vi
end
