################################################################################
# Transport kernel
################################################################################
"""
    monte_carlo_kernel!(E, p_samples, d_samples, R_samples, terminate, props, μ_max, dE, dR_samples, dP_samples, dμ_a, dμ_s, simulation_bounds, spatial_discretication, layout, min_R, propagate_batch, ::Val{K})
CUDA kernel that takes the position and direction arguments of a sample and updates them with `propagate_ray!`.
After `propagate_batch` iterations, `terminate_ray` is called to check whether the ray will continue to the next call of `monte_carlo_kernel!` or not.

`K` is the number of derivative directions tracked simultaneously (`K == 0` disables the
sensitivity computation entirely). It is passed as a `Val` so the tangent state can live in
registers as an `SVector{K,T}` and the per-direction loops unroll at compile time.
"""
function monte_carlo_kernel!(E, p_samples, d_samples, R_samples, terminate,
    props, μ_max,
    dE, dR_samples, dP_samples, dμ_a, dμ_s,
    simulation_bounds, spatial_discretication, layout,
    min_R, propagate_batch, ::Val{K}) where K

    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    @inbounds terminate[idx] && return nothing
    p₁, p₂, p₃ = p_samples[1, idx], p_samples[2, idx], p_samples[3, idx]
    d₁, d₂, d₃ = d_samples[1, idx], d_samples[2, idx], d_samples[3, idx]
    ray_energy = R_samples[idx]
    Td = eltype(dR_samples)
    ∂ray_energy = @per_direction K Td k -> dR_samples[k, idx]
    dP = @per_direction K Td k -> dP_samples[k, idx]

    for _ = 1:propagate_batch
        p₁, p₂, p₃, d₁, d₂, d₃, ray_energy, ∂ray_energy, dP = propagate_ray!(
            E, p₁, p₂, p₃, d₁, d₂, d₃,
            ray_energy, props, μ_max,
            dE, ∂ray_energy, dP, dμ_a, dμ_s,
            spatial_discretication, layout)
        outside_bounds(p₁, p₂, p₃, simulation_bounds) && break
    end

    ray_energy += (ray_energy<min_R && rand()<0.01) * (ray_energy * 99)

    terminate[idx] = terminate_ray(p₁, p₂, p₃, ray_energy, min_R, simulation_bounds)

    p_samples[1, idx], p_samples[2, idx], p_samples[3, idx] = p₁, p₂, p₃
    d_samples[1, idx], d_samples[2, idx], d_samples[3, idx] = d₁, d₂, d₃
    R_samples[idx] = ray_energy
    for k = 1:K
        @inbounds dR_samples[k, idx] = ∂ray_energy[k]
        @inbounds dP_samples[k, idx] = dP[k]
    end

    return nothing
end

"""
    terminate_ray(p1, p2, p3, E_ray, E_threshold, bound)
Checks, whether the coordinates `[p1, p2, p3]` are inside of the simulations bounds set by `bounds`
and if the energy of the ray is higher than `R_threshold`
"""
function terminate_ray(p1, p2, p3, E_ray, E_threshold, bound)
    return E_ray < E_threshold || outside_bounds(p1, p2, p3, bound)
end

"""
    outside_bounds(p1, p2, p3, bound)
Whether `[p1, p2, p3]` has left the simulation box.
"""
@inline function outside_bounds(p1, p2, p3, bound)
    return (p1 < bound[1][1] || p2 < bound[1][2] || p3 < bound[1][3] ||
            p1 > bound[2][1] || p2 > bound[2][2] || p3 > bound[2][3])
end

################################################################################
# Move and scattering
################################################################################
"""
    propagate_ray!(Res, p₁::T, p₂::T, p₃::T, d₁::T, d₂::T, d₃::T, R::T, props, μ_max::T, dRes, dR::SVector{K,T}, dP::SVector{K,T}, dμ_a, dμ_s, spatial_discretication, layout)
Low-level function to perform a single update on the position and direction of a ray.

First, a step length is drawn with a mean free path length of `1/μ_max` and the
ray is propagated that distance in the direction of `[d₁, d₂, d₃]`. The total
interaction coefficient at the new coordinates is then used to sample interaction
and null events. If an interaction occurs, a fraction `μ_a/(μ_a + μ_s)` of the
energy `R` is absorbed and a new propagation direction is sampled. If `K > 0`, the
sensitivity of the result w.r.t. the `K` perturbation directions `dμ_a[k, :]`,
`dμ_s[k, :]` is also computed.

Nothing on the path -- the step length, the interaction/null decision, the scattering
direction -- depends on `dμ_a` or `dμ_s`, so the same path serves all `K` directions and
only the tangent bookkeeping is repeated per `k`. `K` is a type parameter (carried by the
`SVector` tangent state), so the `K == 0` branches vanish at compile time.

The absorbed energy is atomically accumulated into `Res[n]` and its derivative along
direction `k` into `dRes[k, n]`, where `n` is the flat voxel index produced by
[`voxel_index`](@ref).
"""
function propagate_ray!(Res, p₁::T, p₂::T, p₃::T, d₁::T, d₂::T, d₃::T, E_ray::T,
                        props, μ_max::T, dRes, ∂E_ray::SVector{K,T}, dP::SVector{K,T},
                        dμ_a, dμ_s, spatial_discretication, layout) where {T,K}

    Δp = -log(clamp(rand(T), eps(T), one(T))) / μ_max

    p₁ = p₁ + d₁ * Δp
    p₂ = p₂ + d₂ * Δp
    p₃ = p₃ + d₃ * Δp

    # The grid is padded, so index 1 and `layout.dims[k]` are the guard cells a
    # ray that has left the box gets clamped into.
    i = clamp(unsafe_trunc(Int, p₁/spatial_discretication[1] + 1), 0, layout.dims[1] - 1) + 1
    j = clamp(unsafe_trunc(Int, p₂/spatial_discretication[2] + 1), 0, layout.dims[2] - 1) + 1
    l = clamp(unsafe_trunc(Int, p₃/spatial_discretication[3] + 1), 0, layout.dims[3] - 1) + 1
    n = voxel_index(layout, i, j, l)

    # One transaction: the four values are 16 contiguous, aligned bytes.
    @inbounds begin
        μ_a_p = props[1, n]
        μ_s_p = props[2, n]
        g_p   = props[3, n]
        μ_e   = props[4, n]
    end

    if rand(T) < μ_e/μ_max
        d₁, d₂, d₃ = scatter(d₁, d₂, d₃, g_p)
        absorbed_energy = E_ray * μ_a_p / μ_e

        if K > 0
            # `E_ray` and `∂E_ray` are both reassigned further down, so capturing them directly in
            # the `ntuple` closures would box them and destroy inference inside the kernel.
            # These aliases are never reassigned and therefore capture by value.
            _E_ray, _∂E_ray = E_ray, ∂E_ray
            dP += @per_direction K T k -> (dμ_a[k, n] + dμ_s[k, n]) / μ_e
            ∂absorbed_energy = @per_direction K T k -> (_E_ray * (μ_s_p * dμ_a[k, n] - μ_a_p * dμ_s[k, n]) / μ_e^2 +
                                          _∂E_ray[k] * μ_a_p / μ_e)
            for k = 1:K
                @inbounds CUDA.@atomic dRes[k, n] += convert(eltype(dRes),
                    absorbed_energy * dP[k] + ∂absorbed_energy[k])
            end
            ∂E_ray -= ∂absorbed_energy
        end

        CUDA.@atomic Res[n] += convert(eltype(Res), absorbed_energy)
        E_ray -= absorbed_energy
    elseif K > 0
        dP -= @per_direction K T k -> (dμ_a[k, n] + dμ_s[k, n]) / (μ_max - μ_e)
    end
    return p₁, p₂, p₃, d₁, d₂, d₃, E_ray, ∂E_ray, dP
end

"""
    scatter(d₁::T, d₂::T, d₃::T, g::T)
Samples Henyey-Greenstein scattering function with anisotropy `g` for a new ray-direction given the current direction `[d₁, d₂, d₃]`.

!!! note
    The case `d₃ ≈ 1` is handled separately (without branching), as the generic sampling function has a singularity in this case.
"""
function scatter(d₁::T, d₂::T, d₃::T, g::T) where T
    d₁′, d₂′, d₃′ = zero(d₁), zero(d₂), zero(d₃)

    if isapprox(g, one(T))
        cosθ = 1f0
    elseif isapprox(g, zero(T))
        cosθ = 2*rand(T) - 1
    else
        cosθ = (1 + g^2 - ((1 - g^2) / (1 - g + 2 * g * rand(T)))^2) / (2 * g)
    end
    sinθ = sqrt(clamp(1 - cosθ^2, zero(T), one(T)))
    sinϕ, cosϕ = sincospi(2 * rand(T))

    in_axis = isapprox(abs(d₃), 1)
    sqrt1mdz = sqrt(1 - !in_axis * d₃^2)

    d₁′ = in_axis * sinθ * cosϕ     + !in_axis * (sinθ * (d₁ * d₃ * cosϕ - d₂ * sinϕ) / sqrt1mdz + d₁ * cosθ)
    d₂′ = in_axis * sinθ * sinϕ     + !in_axis * (sinθ * (d₂ * d₃ * cosϕ + d₁ * sinϕ) / sqrt1mdz + d₂ * cosθ)
    d₃′ = in_axis * sign(d₃) * cosθ + !in_axis * (-sinθ * cosϕ * sqrt1mdz + d₃ * cosθ)
    
    d_inv_norm = 1/sqrt(d₁′^2 + d₂′^2 + d₃′^2 + eps(T))
    return d₁′*d_inv_norm, d₂′*d_inv_norm, d₃′*d_inv_norm
end
