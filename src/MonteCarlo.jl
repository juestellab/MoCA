################################################################################
#Controll functions
################################################################################
function light_transport_gpu(L₀::BeamSampler{T}, μ_a, μ_s, g, grid_size, simulation_bounds,
                             N, iterations_bunch; n_threads=256, n_blocks=1024, max_loops=Inf, 
                             term_threshold=1f-5) where T
    dμ_a = nothing
    dμ_s = nothing
    return light_transport_gpu(L₀, μ_a, μ_s, dμ_a, dμ_s, g, grid_size, simulation_bounds, N, iterations_bunch;
                               n_threads=n_threads, n_blocks=n_blocks, max_loops=max_loops, 
                               term_threshold=term_threshold)
end
"""
    light_transport_gpu(L₀::BeamSampler{T}, μ_a, μ_s, dμ_a, dμ_s, g, grid_size, simulation_bounds, N, iterations_bunch; n_threads, n_blocks, max_loops)

Compute absorbed light in a 3D volume. The optical properties are specified by μ_a and μ_s, the illumination by L_0.
If 'sensitivity=true', the derivative of the absorbed energy in the direction of (dμ_a, dμ_s) is also tracked. 
The function returns the absorbed energy as a 3D array, as well as the derivative if it is computed.

# Arguments
L₀                      Defines light source. Both spatial distribution of the intensity and distribution over movement-directions of the rays.
μ_a                     3D array with absorption coefficients for each voxel
μ_s                     3D array with scattering coefficients for each voxel
dμ_a                    Infinitesimal change of absorption coefficients for sensitivity computation
dμ_s                    Infinitesimal change of scattering coefficients for sensitivity computation
g                       Scattering anisotropy coefficient, either a scalar or a 3D array with the same shape as μ_a and μ_s
grid_size               Physical dimensions of the grid
simulation_bounds       Corners of box, for which rays will be simulated. If a ray leaves the box, it will no longer be propagated.
N                       Number of rays
iterations_bunch        Number of ray-propagation steps that are performed before checking if any rays have terminated and replacing them.
n_threads               Number of CUDA threads
n_blocks                Number of CUDA blocks
max_loops               Maximum number of ray-propagation steps that will be performed. It it is exceeded,
                        the ray is terminated regardless of its state

# Example
    μ_a = randn(100, 10, 100)
    μ_s = randn(100, 10, 100)
    
    L₀ = BeamSampler(gaussian_beam_sampler((0., 0.)))

    light_transport_gpu(L₀, μ_a, μ_s, 0.9, (1., 0.1, 1.), ((-1., -0.1, 1.), (2., 0.2, 2.)), 100_000, 100)

"""
function light_transport_gpu(L₀::BeamSampler{T}, μ_a, μ_s, dμ_a, dμ_s, g, grid_size, simulation_bounds,
                             N, iterations_bunch; n_threads=256, n_blocks=1024, max_loops=Inf, 
                             term_threshold=1f-5) where T
    @assert size(μ_a) == size(μ_s)
    xor(isnothing(dμ_a), isnothing(dμ_s)) && error("Either both or neither dμ_a and dμ_s can be nothing.")
    sensitivity = !(isnothing(dμ_a) && isnothing(dμ_s))
    println("sensitivity=$(sensitivity)")

    spatial_discretization = SA{T}[(grid_size ./ size(μ_a))...]
    println("dspace = $(spatial_discretization)")
    simulation_bounds = Tuple(SA{T}[bounds...] for bounds in simulation_bounds)
    println("sim bounds = $(simulation_bounds)")
    μ_max = T(maximum(μ_a .+ μ_s) * (sensitivity ? 1.2 : 1))
    println("μ_max=$(μ_max) max(μ_a)=$(maximum(μ_a)) max(μ_s)=$(maximum(μ_s)) min(g)=$(minimum(g))")
    if isa(g, Number)
        g = fill(T(g), size(μ_a))
    end
    @assert maximum(abs.(g)) <= 1 "Anisotropy must be in the interval [-1, 1]"
    n_threads, n_blocks, n_batch = cuda_params(N, n_threads, n_blocks)
    println("threads=$n_threads blocks=$n_blocks batches=$n_batch")

    μ_a, μ_s, dμ_a, dμ_s, g = map(x -> isnothing(x) ? CUDA.zeros(T, 1) : CuArray{T}(pad_array(x)), (μ_a, μ_s, dμ_a, dμ_s, g))
    E = CUDA.zeros(T, size(μ_a))
    dE = sensitivity ? zero(E) : CUDA.zeros(T, 1)

    CUDA.@sync p_samples, d_samples, R_samples, dR_samples, dP_samples = initialize_samples(L₀, n_batch)
    terminate = CUDA.fill(false, n_batch)

    p = Progress(N, dt=0.5, showspeed=true)
    loops = 0
    n_finished_rays = 0
    @inline callback!(Res, _ ,_, _, _, _, _, _, _, W, i, j, l) = (CUDA.@atomic Res[i, j, l] += W)
    eval_cuda_kernel!(E, p_samples, d_samples, R_samples, terminate, dE, dR_samples, dP_samples) = @cuda threads = n_threads blocks = n_blocks monte_carlo_kernel!(
        E, p_samples, d_samples, R_samples, terminate,
        μ_a, μ_s, g, μ_max,
        dE, dR_samples, dP_samples, dμ_a, dμ_s,
        simulation_bounds, spatial_discretization,
        T(term_threshold),
        iterations_bunch, callback!, sensitivity)
    @info "Start propagating $N rays"
    while n_finished_rays < N && loops < max_loops
        Δt = @timed begin
            eval_cuda_kernel!(E, p_samples, d_samples, R_samples, terminate, dE, dR_samples, dP_samples)
            CUDA.@sync blocking=false begin 
                p_samples, d_samples, R_samples, dR_samples, dP_samples = refill_samples!(
                    L₀, p_samples, d_samples, R_samples, dR_samples, dP_samples, terminate)
            end 
        end
        n_finished_rays += sum(terminate)
        loops += n_threads * n_blocks * iterations_bunch
        update!(p, n_finished_rays, 
                showvalues=[(Symbol("Rays/s"), sum(terminate)/Δt.time), 
                            (Symbol("Energy"), "$(100*sum(E)/n_finished_rays)%")])
        terminate = CUDA.fill(false, n_batch)
    end

    loops >= max_loops && @warn "Some rays were terminated as they exceeded the `max_loops` threshold"
    n_finished_rays += sum(terminate)
    println("\n\n")
    @info("Total energy: $(100*sum(E)/n_finished_rays)%")
    E = unpad(E)./n_finished_rays
    if sensitivity
        return E, unpad(dE) ./ n_finished_rays
    else
        return E
    end
end

"""
    initialize_samples(beam_sampler, n_samples) 
Draw positions, directions and energies for `n_samples` rays and initialize the differential parameters
"""
function initialize_samples(beam_sampler, n_samples)
    p_samples, d_samples, R_samples = next(beam_sampler, n_samples)
    T = typeof(beam_sampler).parameters[1]
    return map(CuArray, (p_samples, d_samples, R_samples, zeros(T, n_samples), zeros(T, n_samples)))
end


"""
    refill_samples!(beam_sampler::BeamSampler{T}, p, d, R, dR, dP, terminate)
If an entry in `terminate` is true, replace the corresponding entries in
`p`, `d` with a new sample and reset `R`, `dR`, `dP` to zero.
"""
function refill_samples!(beam_sampler::BeamSampler{T}, p, d, R, dR, dP, terminate) where T
    p_CPU, d_CPU, R_CPU, dR_CPU, dP_CPU, terminate_CPU = map(Array, (p, d, R, dR, dP, terminate))
    # sum up terminated energy as consistency check

    n_new_samples = Int(sum(terminate_CPU))
    # @assert n_new_samples > 0 "n_new_samples = $(n_new_samples)"
    new_p, new_d, new_R = next(beam_sampler, n_new_samples)

    p_CPU[:, terminate_CPU] .= new_p
    d_CPU[:, terminate_CPU] .= new_d
    R_CPU[terminate_CPU] .= new_R
    dR_CPU[terminate_CPU] .= zero(T)
    dP_CPU[terminate_CPU] .= zero(T)

    return map(CuArray, (p_CPU, d_CPU, R_CPU, dR_CPU, dP_CPU))
end


function split_samples!(p, d, R, dR, dP, terminate)
    p_CPU, d_CPU, R_CPU, dR_CPU, dP_CPU, terminate_CPU = map(Array, (p, d, R, dR, dP, terminate))

    terminate_idx = findall(terminate_CPU)
    continue_idx = findall(!, terminate_CPU)
    for i = 1:min(length(terminate_idx), length(continue_idx))
        p_CPU[:, terminate_idx[i]] .= p_CPU[:, continue_idx[i]]
        d_CPU[:, terminate_idx[i]] .= d_CPU[:, continue_idx[i]]
        R_CPU[terminate_idx[i]] = R_CPU[continue_idx[i]]/2
        dR_CPU[terminate_idx[i]] = dR_CPU[continue_idx[i]]/2
        dP_CPU[terminate_idx[i]] = dP_CPU[continue_idx[i]]
        terminate_CPU[terminate_idx[i]] = false

        R_CPU[continue_idx[i]] = R_CPU[continue_idx[i]]/2
        dR_CPU[continue_idx[i]] = dR_CPU[continue_idx[i]]/2
    end

    return map(CuArray, (p_CPU, d_CPU, R_CPU, dR_CPU, dP_CPU, terminate_CPU))
end


"""
    monte_carlo_kernel!(E, p_samples, d_samples, R_samples, terminate, μ_a, μ_s, g, μ_max, dE, dR_samples, dP_samples, dμ_a, dμ_s, simulation_bounds, spatial_discretication, min_R, propagate_batch, callback!, sensitivity)
CUDA kernel that takes the position and direction arguments of a sample and updates them with `propagate_ray!`.
After `propagate_batch` iterations, `terminate_ray` is called to check whether the ray will continue to the next call of `monte_carlo_kernel!` or not.
"""
function monte_carlo_kernel!(E, p_samples, d_samples, R_samples, terminate,
    μ_a, μ_s, g, μ_max,
    dE, dR_samples, dP_samples, dμ_a, dμ_s,
    simulation_bounds, spatial_discretication,
    min_R, propagate_batch, callback!, sensitivity)

    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    p₁, p₂, p₃ = p_samples[1, idx], p_samples[2, idx], p_samples[3, idx]
    d₁, d₂, d₃ = d_samples[1, idx], d_samples[2, idx], d_samples[3, idx]
    R = R_samples[idx]
    dR = dR_samples[idx]
    dP = dP_samples[idx]

    for _ = 1:propagate_batch
        p₁, p₂, p₃, d₁, d₂, d₃, R, dR, dP = propagate_ray!(
            E, p₁, p₂, p₃, d₁, d₂, d₃,
            R, μ_a, μ_s, μ_max, g,
            dE, dR, dP, dμ_a, dμ_s,
            spatial_discretication, callback!, sensitivity)
    end

    R += (R<min_R && rand()<0.01) * (R * 99)

    terminate[idx] = terminate_ray(p₁, p₂, p₃, R, min_R, simulation_bounds)

    p_samples[1, idx], p_samples[2, idx], p_samples[3, idx] = p₁, p₂, p₃
    d_samples[1, idx], d_samples[2, idx], d_samples[3, idx] = d₁, d₂, d₃
    R_samples[idx] = R
    dR_samples[idx] = dR
    dP_samples[idx] = dP

    return nothing
end

"""
    terminate_ray(p1, p2, p3, R, R_threshold, bound)
Checks, whether the coordinates `[p1, p2, p3]` are inside of the simulations bounds set by `bounds`
and if the energy of the ray is higher than `R_threshold`
"""
function terminate_ray(p1, p2, p3, R, R_threshold, bound)
    return (R < R_threshold ||
            p1 < bound[1][1] || p2 < bound[1][2] || p3 < bound[1][3] ||
            p1 > bound[2][1] || p2 > bound[2][2] || p3 > bound[2][3])
end


################################################################################
#Move and Scattering
################################################################################
"""
    propagate_ray!(Res, p₁::T, p₂::T, p₃::T, d₁::T, d₂::T, d₃::T, R::T, μ_a, μ_s, μ_max::T, g, dRes, dR::T, dP::T, dμ_a, dμ_s, spatial_discretication, callback!, sensitivity)
Low-level function to perform a single update on the position and direction of a ray.

First, a step length is drawn with a mean free path length of `1/μ_max` and the
ray is propagated that distance in the direction of `[d₁, d₂, d₃]`. The total
interaction coefficient at the new coordinates is then used to sample interaction
and null events. If an interaction occurs, a fraction `μ_a/(μ_a + μ_s)` of the
energy `R` is absorbed and a new propagation direction is sampled. If 
`sensitivity == true`, the sensitivity of the result w.r.t changes `dμ_a`, `dμ_s`
is also computed. 

The `callback!` function is used to track the quantity of interest.
The passed arguments are result-array, the three position and three direction variables,
the step size, change in energy and array-index of the ray position.
"""
function propagate_ray!(Res, p₁::T, p₂::T, p₃::T, d₁::T, d₂::T, d₃::T, R::T,
                        μ_a, μ_s, μ_max::T, g,
                        dRes, dR::T, dP::T, dμ_a, dμ_s,
                        spatial_discretication, callback!, sensitivity) where T

    Δp = -log(clamp(rand(T), eps(T), one(T))) / μ_max
    p₁ = p₁ + d₁ * Δp
    p₂ = p₂ + d₂ * Δp
    p₃ = p₃ + d₃ * Δp

    i = clamp(unsafe_trunc(Int, p₁/spatial_discretication[1] + 1), 0, size(Res, 1) - 1) + 1 
    j = clamp(unsafe_trunc(Int, p₂/spatial_discretication[2] + 1), 0, size(Res, 2) - 1) + 1
    l = clamp(unsafe_trunc(Int, p₃/spatial_discretication[3] + 1), 0, size(Res, 3) - 1) + 1

    μ_a_p = μ_a[i, j, l]
    μ_s_p = μ_s[i, j, l]
    g_p = g[i, j, l]
    μ_e = μ_a_p + μ_s_p

    if sensitivity
        dμ_a_p = dμ_a[i, j, l]
        dμ_s_p = dμ_s[i, j, l]
    end

    if rand(T) < μ_e/μ_max
        d₁, d₂, d₃ = scatter(d₁, d₂, d₃, g_p)
        W = R * μ_a_p / μ_e

        if sensitivity
            dP += (dμ_a_p + dμ_s_p) / μ_e
            
            dW = R * (μ_s_p * dμ_a_p - μ_a_p * dμ_s_p) / μ_e^2  + dR*μ_a_p/μ_e
            dE = W * dP + dW
            dR -= dW
            callback!(dRes, p₁, p₂, p₃, d₁, d₂, d₃, R, Δp, dE, i, j, l)
        end

        R -= W
        callback!(Res, p₁, p₂, p₃, d₁, d₂, d₃, R, Δp, W, i, j, l)
    elseif sensitivity
        dP -= (dμ_a_p + dμ_s_p) / (μ_max - μ_e)
    end
    return p₁, p₂, p₃, d₁, d₂, d₃, R, dR, dP
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

    d₁′ = in_axis * sinθ * cosϕ + !in_axis * (sinθ * (d₁ * d₃ * cosϕ - d₂ * sinϕ) / sqrt1mdz + d₁ * cosθ)
    d₂′ = in_axis * sinθ * sinϕ + !in_axis * (sinθ * (d₂ * d₃ * cosϕ + d₁ * sinϕ) / sqrt1mdz + d₂ * cosθ)
    d₃′ = in_axis * cosθ        + !in_axis * (-sinθ * cosϕ * sqrt1mdz + d₃ * cosθ)
    
    d_inv_norm = 1/sqrt(d₁′^2 + d₂′^2 + d₃′^2 + eps(T))
    return d₁′*d_inv_norm, d₂′*d_inv_norm, d₃′*d_inv_norm
end
