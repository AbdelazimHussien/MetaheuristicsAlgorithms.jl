"""
    THRO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)

Tianji's Horse Racing Optimization (THRO).
Maintains two equal-size subpopulations (Tianji and King) in column-major matrices `dim×(npop/2)` and evolves them via comparison-driven moves and Lévy disturbances.

Arguments
- `objfun(x::AbstractVector) -> Real` – objective to minimize.
- `lb::Vector{Float64}` – lower bounds (length = dim).
- `ub::Vector{Float64}` – upper bounds (length = dim).
- `npop::Integer` – total individuals (even; split equally).
- `max_iter::Integer` – maximum iterations.

Returns
- `OptimizationResult` – a result object containing:
  - best fitness (minimum objective value)
  - best solution vector
  - history of best fitness per iteration

Notes
- Uses in-place shuffling, sorting, elite-guided local search, and Lévy-flight perturbations (`levy_mantegna`).
- Bounds are enforced with `spacebound_reinit!`.

Reference
Wang et al., "Tianji’s horse racing optimization (THRO): a new metaheuristic inspired by ancient wisdom and its engineering optimization applications.", Artificial Intelligence Review 58(9), 2025.
"""
function THRO(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    @assert iseven(npop) "npop must be even (sum of two equal populations)."
    rng = Random.GLOBAL_RNG
    nside = Int(npop ÷ 2)

    dim = length(lb)
    Δ   = ub - lb
    T   = eltype(lb)
    w   = fill(T(1 / nside), nside)

    tianji_x = Matrix{T}(undef, dim, nside)
    king_x   = Matrix{T}(undef, dim, nside)
    best_x   = Vector{T}(undef, dim)
    rand!(rng, tianji_x)
    @. tianji_x = lb + tianji_x * Δ
    rand!(rng, king_x)
    @. king_x = lb + king_x * Δ

    tianji_y = Vector{T}(undef, nside)
    king_y   = Vector{T}(undef, nside)
    best_y   = typemax(T)

    @views for i = 1:nside
        tianji_y[i] = objfun(tianji_x[:, i])
        king_y[i] = objfun(king_x[:, i])
    end

    t_best_y, t_idx = findmin(tianji_y)
    k_best_y, k_idx = findmin(king_y)
    if t_best_y < k_best_y
        best_y = t_best_y
        best_x .= view(tianji_x, :, t_idx)
    else
        best_y = k_best_y
        best_x .= view(king_x, :, k_idx)
    end

    hist_best_y = fill(best_y, max_iter)

    # temporary buffer: reuse for shuffling/reordering/sorting, avoid new allocation
    perm0       = Vector{Int}(undef, npop)
    perm_dim    = Vector{Int}(undef, dim)
    perm_side   = Vector{Int}(undef, nside)
    tianji_xnew = Vector{T}(undef, dim)
    tianji_xbuf = similar(tianji_x)      # dim×npop
    tianji_ybuf = similar(tianji_y)      # nside
    king_xnew   = Vector{T}(undef, dim)
    king_xbuf   = similar(king_x)
    king_ybuf   = similar(king_y)
    t_snap      = Vector{T}(undef, dim)  # tianji's slowest horse snapshot
    t_fsnap     = Vector{T}(undef, dim)  # tianji's fastest horse snapshot
    r_t         = Vector{T}(undef, dim)  # Lévy disturbance (tianji)
    r_k         = Vector{T}(undef, dim)  # Lévy disturbance (king)
    mean_t      = Vector{T}(undef, dim)
    mean_k      = Vector{T}(undef, dim)

    for iter = 1:max_iter
        randperm!(rng, perm0)
        @inbounds @views for j = 1:nside
            s = perm0[j] # To tianji
            if s <= nside
                tianji_xbuf[:, j] .= tianji_x[:, s]
                tianji_ybuf[j] = tianji_y[s]
            else
                s -= nside
                tianji_xbuf[:, j] .= king_x[:, s]
                tianji_ybuf[j] = king_y[s]
            end
            s = perm0[nside+j] # To king
            if s <= nside
                king_xbuf[:, j] .= tianji_x[:, s]
                king_ybuf[j] = tianji_y[s]
            else
                s -= nside
                king_xbuf[:, j] .= king_x[:, s]
                king_ybuf[j] = king_y[s]
            end
        end
        # swap binding: use buffer as new population (zero allocation, zero copy)
        tianji_x, tianji_xbuf = tianji_xbuf, tianji_x
        king_x, king_xbuf     = king_xbuf, king_x
        tianji_y, tianji_ybuf = tianji_ybuf, tianji_y
        king_y, king_ybuf     = king_ybuf, king_y

        # sort each population by fitness; reorder columns with y (reuse buffer)
        # Tianji
        sortperm!(perm_side, tianji_y)
        @inbounds @views for j = 1:nside
            pj                = perm_side[j]
            tianji_ybuf[j]    = tianji_y[pj]
            tianji_xbuf[:, j] .= tianji_x[:, pj]
        end
        tianji_y, tianji_ybuf = tianji_ybuf, tianji_y # swap binding
        tianji_x, tianji_xbuf = tianji_xbuf, tianji_x
        # King
        sortperm!(perm_side, king_y)
        @inbounds @views for j = 1:nside
            pk              = perm_side[j]
            king_ybuf[j]    = king_y[pk]
            king_xbuf[:, j] .= king_x[:, pk]
        end
        king_y, king_ybuf = king_ybuf, king_y
        king_x, king_xbuf = king_xbuf, king_x

        p = 1 - (iter / max_iter)
        t_fastest_id, t_slowest_id = 1, nside
        k_fastest_id, k_slowest_id = 1, nside
        t_1st = @view tianji_x[:, 1]
        k_1st = @view king_x[:, 1]

        @views for i = 1:nside
            t_alpha = 1 + (rand(rng) < 0.5 ? randn(rng) : 0.0) # ~50%
            t_beta  = (rand(rng) < 0.1 ? randn(rng) : 0.0)     # ~10%
            k_alpha = 1 + (rand(rng) < 0.5 ? randn(rng) : 0.0)
            k_beta  = (rand(rng) < 0.1 ? randn(rng) : 0.0)

            # Lévy disturbance (Eq. 7): construct sparse r_t / r_k for each individual (zero allocation)
            levy_t = levy_mantegna(; rng)
            levy_k = levy_mantegna(; rng)
            fill!(r_t, 0)
            randperm!(rng, perm_dim)
            rnum = ceil(Int, sinpi(0.5rand(rng)) * dim)
            @inbounds for k = 1:rnum
                r_t[perm_dim[k]] = levy_t
            end
            fill!(r_k, 0)
            randperm!(rng, perm_dim)
            rnum = ceil(Int, sinpi(0.5rand(rng)) * dim)
            @inbounds for k = 1:rnum
                r_k[perm_dim[k]] = levy_k
            end
            # first mean (before any write back in this round; for tianji's proposal)
            mul!(mean_t, tianji_x, w)
            mul!(mean_k, king_x, w)

            if tianji_y[t_slowest_id] < king_y[k_slowest_id]
                # [scenario 1] (Eq. 3 & 14)
                t_snap .= tianji_x[:, t_slowest_id]  # slowest horse snapshot before update

                @. tianji_xnew = ((p * t_snap + (1 - p) * t_1st) +
                                  r_t * (t_1st - t_snap + p * (mean_t - mean_k))) * t_alpha + t_beta
                spacebound_reinit!(tianji_xnew, ub, lb; rng=rng)
                t_y_new = objfun(tianji_xnew)
                if t_y_new < tianji_y[t_slowest_id]
                    tianji_y[t_slowest_id] = t_y_new
                    tianji_x[:, t_slowest_id] .= tianji_xnew
                end

                # second mean (tianji may write back after this round; for king's proposal)
                mul!(mean_t, tianji_x, w)
                mul!(mean_k, king_x, w)

                k_slowest = view(king_x, :, k_slowest_id)
                @. king_xnew = ((p * k_slowest + (1 - p) * t_snap) +
                                r_k * (t_snap - k_slowest + p * (mean_t - mean_k))) * k_alpha + k_beta
                spacebound_reinit!(king_xnew, ub, lb; rng=rng)
                k_y_new = objfun(king_xnew)
                if k_y_new < king_y[k_slowest_id]
                    king_y[k_slowest_id] = k_y_new
                    king_x[:, k_slowest_id] .= king_xnew
                end

                t_slowest_id -= 1
                k_slowest_id -= 1
            elseif tianji_y[t_slowest_id] > king_y[k_slowest_id]
                # [scenario 2] (Eq. 15 & 16)
                tr = rand(rng, 1:nside)
                t_peer = view(tianji_x, :, tr)
                t_snap .= tianji_x[:, t_slowest_id]

                @. tianji_xnew = ((p * t_snap + (1 - p) * t_peer) +
                                  r_t * (t_peer - t_snap + p * (mean_t - mean_k))) * t_alpha + t_beta
                spacebound_reinit!(tianji_xnew, ub, lb; rng=rng)
                t_y_new = objfun(tianji_xnew)
                if t_y_new < tianji_y[t_slowest_id]
                    tianji_y[t_slowest_id] = t_y_new
                    tianji_x[:, t_slowest_id] .= tianji_xnew
                end

                mul!(mean_t, tianji_x, w)
                mul!(mean_k, king_x, w)

                k_fastest = view(king_x, :, k_fastest_id)
                @. king_xnew = ((p * k_fastest + (1 - p) * k_1st) +
                                r_k * (k_1st - k_fastest + p * (mean_t - mean_k))) * k_alpha + k_beta
                spacebound_reinit!(king_xnew, ub, lb; rng=rng)
                k_y_new = objfun(king_xnew)
                if k_y_new < king_y[k_fastest_id]
                    king_y[k_fastest_id] = k_y_new
                    king_x[:, k_fastest_id] .= king_xnew
                end

                t_slowest_id -= 1
                k_fastest_id += 1
            else # slow horse equal → see fastest horse（3/4/5）
                if tianji_y[t_fastest_id] < king_y[k_fastest_id]
                    # [scenario 3] (Eq. 17 & 18)
                    t_fsnap .= tianji_x[:, t_fastest_id]

                    @. tianji_xnew = ((p * t_fsnap + (1 - p) * t_1st) +
                                      r_t * (t_1st - t_fsnap + p * (mean_t - mean_k))) * t_alpha + t_beta
                    spacebound_reinit!(tianji_xnew, ub, lb; rng=rng)
                    t_y_new = objfun(tianji_xnew)
                    if t_y_new < tianji_y[t_fastest_id]
                        tianji_y[t_fastest_id] = t_y_new
                        tianji_x[:, t_fastest_id] .= tianji_xnew
                    end

                    mul!(mean_t, tianji_x, w)
                    mul!(mean_k, king_x, w)

                    k_fastest = view(king_x, :, k_fastest_id)
                    @. king_xnew = ((p * k_fastest + (1 - p) * t_fsnap) +
                                    r_k * (t_fsnap - k_fastest + p * (mean_t - mean_k))) * k_alpha + k_beta
                    spacebound_reinit!(king_xnew, ub, lb; rng=rng)
                    k_y_new = objfun(king_xnew)
                    if k_y_new < king_y[k_fastest_id]
                        king_y[k_fastest_id] = k_y_new
                        king_x[:, k_fastest_id] .= king_xnew
                    end

                    t_fastest_id += 1
                    k_fastest_id += 1
                else
                    # [scenario 4/5] (Eq. 19–22)
                    tr = rand(rng, 1:nside)
                    t_peer = view(tianji_x, :, tr)
                    t_snap .= tianji_x[:, t_slowest_id]

                    @. tianji_xnew = ((p * t_snap + (1 - p) * t_peer) +
                                      r_t * (t_peer - t_snap + p * (mean_t - mean_k))) * t_alpha + t_beta
                    spacebound_reinit!(tianji_xnew, ub, lb; rng=rng)
                    t_y_new = objfun(tianji_xnew)
                    if t_y_new < tianji_y[t_slowest_id]
                        tianji_y[t_slowest_id] = t_y_new
                        tianji_x[:, t_slowest_id] .= tianji_xnew
                    end

                    mul!(mean_t, tianji_x, w)
                    mul!(mean_k, king_x, w)

                    k_fastest = view(king_x, :, k_fastest_id)
                    @. king_xnew = ((p * k_fastest + (1 - p) * k_1st) +
                                    r_k * (k_1st - k_fastest + p * (mean_t - mean_k))) * k_alpha + k_beta
                    spacebound_reinit!(king_xnew, ub, lb; rng=rng)
                    k_y_new = objfun(king_xnew)
                    if k_y_new < king_y[k_fastest_id]
                        king_y[k_fastest_id] = k_y_new
                        king_x[:, k_fastest_id] .= king_xnew
                    end

                    t_slowest_id -= 1
                    k_fastest_id += 1
                end
            end
        end

        # ---- Track current global best（once; tie preserved for king）----
        t_best_y, t_idx = findmin(tianji_y)
        k_best_y, k_idx = findmin(king_y)
        if t_best_y < k_best_y
            best_y = t_best_y
            best_x .= @view tianji_x[:, t_idx]
        else
            best_y = k_best_y
            best_x .= @view king_x[:, k_idx]
        end

        # ---- Elite-guided local search (Eqs. 23–26) ----
        t_best_col = @view tianji_x[:, t_idx]
        k_best_col = @view king_x[:, k_idx]
        decay = 0.001 * (1 - iter / max_iter)^2

        @inbounds @views for i = 1:nside
            # ---- Tianji side ----
            for j = 1:dim
                if rand(rng) > 0.5
                    # differential neighborhood: draw two different indices (lightweight alternative to randperm(...)[1:2])
                    tr4 = rand(rng, 1:nside)
                    r = rand(rng, 1:nside-1)
                    tr5 = (r < tr4) ? r : (r + 1)   # offset mapping
                    LT = 0.2 * levy_mantegna(; rng)
                    tianji_xnew[j] = tianji_x[j, i] + LT * (tianji_x[j, tr4] - tianji_x[j, tr5]) # Eq.24
                else
                    MT = 0.5 * (1 + decay * sinpi(rand(rng))) # Eq.23
                    tb = t_best_col[j]
                    tianji_xnew[j] = tb + MT * (tb - tianji_x[j, i])
                end
            end
            # ---- King side ----
            for j = 1:dim
                if rand(rng) > 0.5
                    kr1 = rand(rng, 1:nside)
                    r = rand(rng, 1:nside-1)
                    kr2 = (r < kr1) ? r : (r + 1)
                    LK = 0.2 * levy_mantegna(; rng)
                    king_xnew[j] = king_x[j, i] + LK * (king_x[j, kr1] - king_x[j, kr2]) # Eq.26
                else
                    MK = 0.5 * (1 + decay * sinpi(rand(rng))) # Eq.25
                    kb = k_best_col[j]
                    king_xnew[j] = kb + MK * (kb - king_x[j, i])
                end
            end

            spacebound_reinit!(tianji_xnew, ub, lb; rng)
            spacebound_reinit!(king_xnew, ub, lb; rng)
            t_y_new = objfun(tianji_xnew)
            k_y_new = objfun(king_xnew)

            if t_y_new < tianji_y[i]
                tianji_y[i] = t_y_new
                tianji_x[:, i] .= tianji_xnew
                if t_y_new < best_y
                    best_y = t_y_new
                    copy!(best_x, tianji_xnew)
                end
            end
            if k_y_new < king_y[i]
                king_y[i] = k_y_new
                king_x[:, i] .= king_xnew
                if k_y_new < best_y
                    best_y = k_y_new
                    best_x .= king_xnew
                end
            end
        end
        hist_best_y[iter] = best_y
    end

    return OptimizationResult(best_x, best_y, hist_best_y)
end

function levy_mantegna(; β::T=1.5, rng=Random.GLOBAL_RNG) where {T<:AbstractFloat}
    b = β
    s = (gamma(1 + b) * sinpi(b / 2) / (gamma((1 + b) / 2) * b * exp2((b - 1) / 2)))^(one(T) / b)
    invb = one(T) / b
    u = s * randn(rng, T)
    v = randn(rng, T)
    return u / abs(v)^invb
end

function levy_mantegna(d::Integer; β::T=1.5, rng=Random.GLOBAL_RNG) where {T<:AbstractFloat}
    b = β
    s = (gamma(1 + b) * sinpi(b / 2) / (gamma((1 + b) / 2) * b * exp2((b - 1) / 2)))^(one(T) / b)
    invb = one(T) / b
    u = s .* randn(rng, T, d)
    v = randn(rng, T, d)
    return @. u / abs(v)^invb
end

function levy_mantegna!(out::AbstractVector{T}; β::T=1.5, rng=Random.GLOBAL_RNG) where {T<:AbstractFloat}
    b = β
    s = (gamma(1 + b) * sinpi(b / 2) / (gamma((1 + b) / 2) * b * exp2((b - 1) / 2)))^(one(T) / b)
    invb = one(T) / b
    @inbounds for i in eachindex(out)
        u = s * randn(rng, T)
        v = randn(rng, T)
        out[i] = u / abs(v)^invb
    end
    return out
end

function spacebound_reinit!(x::AbstractVector{T}, ub::T, lb::T; rng=Random.GLOBAL_RNG) where {T<:AbstractFloat}
    Δ = ub - lb
    @inbounds @simd for i in eachindex(x)
        xi = x[i]
        if (xi > ub) | (xi < lb)
            x[i] = muladd(Δ, rand(rng, T), lb)
        end
    end
    return x
end

function spacebound_reinit!(x::AbstractVector{T}, ub::AbstractVector{T}, lb::AbstractVector{T}; rng=Random.GLOBAL_RNG) where {T<:AbstractFloat}
    @inbounds @simd for i in eachindex(x, ub, lb)
        xi = x[i]
        lbi, ubi = lb[i], ub[i]
        if (xi > ubi) | (xi < lbi)
            x[i] = muladd(ubi - lbi, rand(rng, T), lbi)
        end
    end
    return x
end
