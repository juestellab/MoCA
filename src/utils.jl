"""
    @per_direction K T k -> expr

Build the `SVector{K,T}` whose `k`-th entry is `expr`.

The tangent state of the sensitivity computation is carried in registers as an
`SVector{K}`, so every per-direction quantity has to be written as

    SVector{K,T}(ntuple(k -> ..., Val(K)))

The `Val(K)` is what unrolls the loop at compile time and keeps the vector out of
global memory, so it cannot be dropped.

The index is the parameter of passed lambda. Its body is evaluated inside
`@inbounds`, since every caller here indexes with a loop variable that is
`1:K` by construction.

    dP += @per_direction K T k -> (dμ_a[k, n] + dμ_s[k, n]) / μ_e
"""
macro per_direction(K, T, f)
    (f isa Expr && f.head === :->) || throw(ArgumentError(
        "@per_direction expects a `k -> expr` lambda as its third argument, got `$f`"))
    index, body = f.args
    checked = Expr(:->, index, Expr(:macrocall, Symbol("@inbounds"), __source__, body))
    return :(SVector{$(esc(K)),$(esc(T))}(ntuple($(esc(checked)), Val($(esc(K))))))
end

"""
    normalize_directions(μ_a, dμ_a, dμ_s) -> (dμ_a, dμ_s, K, stacked)

Canonicalise the sensitivity inputs to the `(K, size(μ_a)...)` layout the
transport kernels expect.

`dμ_a`/`dμ_s` may be `nothing` (no sensitivity, `K == 0`), a 3D array shaped like
`μ_a` (one direction, reported back as 3D — `stacked == false`), or a 4D
`(K, size(μ_a)...)` array. Pair with [`denormalize_directions`](@ref) on the way
out.

This is shared by every transport entry point so that what counts as a legal
perturbation shape is defined once.
"""
function normalize_directions(μ_a, dμ_a, dμ_s)
    xor(isnothing(dμ_a), isnothing(dμ_s)) &&
        error("Either both or neither dμ_a and dμ_s can be nothing.")
    isnothing(dμ_a) && return nothing, nothing, 0, false

    @assert size(dμ_a) == size(dμ_s) "dμ_a and dμ_s must have the same shape"
    @assert ndims(dμ_a) in (3, 4) "dμ_a must be 3D (one direction) or 4D (K, size(μ_a)...)"
    @assert size(dμ_a)[end-2:end] == size(μ_a) "trailing dimensions of dμ_a must match size(μ_a)"

    stacked = ndims(dμ_a) == 4
    dμ_a = stacked ? dμ_a : reshape(dμ_a, 1, size(dμ_a)...)
    dμ_s = stacked ? dμ_s : reshape(dμ_s, 1, size(dμ_s)...)
    return dμ_a, dμ_s, size(dμ_a, 1), stacked
end

"""
    denormalize_directions(dE, stacked)

Undo [`normalize_directions`](@ref) on the result: a 3D perturbation went in, so
a 3D derivative comes back out.
"""
denormalize_directions(dE, stacked) = stacked ? dE : dropdims(dE, dims=1)

"""
    cuda_params(N, n_threads, n_blocks)
For `N` total kernel evaluations and a target of `n_threads` threads and `n_blocks` blocks, compute the number of batches required to reach at least `N` kernel evaluations.
In case `n_threads * n_blocks > N`, those parameters are adapted to evaluate only the smallest number of required blocks.
"""
function cuda_params(N, n_threads, n_blocks)
    if N < n_threads * n_blocks 
        n_threads = (N < n_threads) ? N : n_threads
        n_blocks = N ÷ n_threads
    end
    n_batch = n_threads * n_blocks
    return n_threads, n_blocks, n_batch
end
