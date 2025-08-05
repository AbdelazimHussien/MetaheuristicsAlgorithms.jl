initialization(npop::Integer, dim::Integer, ub::Union{Real,AbstractVector{<:Real}}, lb::Union{Real,AbstractVector{<:Real}}) = rand(npop, dim) .* (ub .- lb)' .+ lb'

initialization2(npop::Integer, dim::Integer, ub::Union{Real,AbstractVector{<:Real}}, lb::Union{Real,AbstractVector{<:Real}}) = rand(dim, npop) .* (ub .- lb) .+ lb

initialization2!(x, ub, lb) = @. x = (ub - lb) * $rand!(x) + lb