################################################################################
# Transport driver
################################################################################

function light_transport_gpu(L₀::BeamSampler{T}, μ_a, μ_s, g, grid_size, simulation_bounds,
                             N, iterations_bunch; kwargs...) where T
    return light_transport_gpu(L₀, μ_a, μ_s, nothing, nothing, g, grid_size, simulation_bounds,
                               N, iterations_bunch; kwargs...)
end
"""
    light_transport_gpu(L₀::BeamSampler{T}, μ_a, μ_s, dμ_a, dμ_s, g, grid_size, simulation_bounds, N, iterations_bunch; n_threads, n_blocks, max_loops)

Compute absorbed light in a 3D volume. The optical properties are specified by μ_a and μ_s, the illumination by L_0.
If 'sensitivity=true', the derivative of the absorbed energy in the direction of (dμ_a, dμ_s) is also tracked. 
The function returns the absorbed energy as a 3D array, as well as the derivative if it is computed.
The derivative has the same rank as `dμ_a`: a 3D array for a single direction, or a 4D array
`(K, size(μ_a)...)` when `K` directions are passed.

# Arguments
L₀                      Defines light source. Both spatial distribution of the intensity and distribution over movement-directions of the rays.
μ_a                     3D array with absorption coefficients for each voxel
μ_s                     3D array with scattering coefficients for each voxel
dμ_a                    Infinitesimal change of absorption coefficients for sensitivity computation.
                        Either a 3D array shaped like `μ_a` (a single derivative direction), or a 4D
                        array `(K, size(μ_a)...)` holding `K` directions at once. The ray paths do not
                        depend on the perturbation, so all `K` derivatives share one path ensemble and
                        cost far less than `K` separate runs.
dμ_s                    Infinitesimal change of scattering coefficients for sensitivity computation.
                        Must have the same shape as `dμ_a`.
g                       Scattering anisotropy coefficient, either a scalar or a 3D array with the same shape as μ_a and μ_s
grid_size               Physical dimensions of the grid
simulation_bounds       Corners of box, for which rays will be simulated. If a ray leaves the box, it will no longer be propagated.
N                       Number of rays
iterations_bunch        Number of ray-propagation steps that are performed before checking if any rays have terminated and replacing them.
n_threads               Number of CUDA threads
n_blocks                Number of CUDA blocks
max_loops               Maximum number of ray-propagation steps that will be performed. It it is exceeded,
                        the ray is terminated regardless of its state
drain                   Finish the rays still in flight when the main loop exits, rather than
                        abandoning them (default `true`). The loop stops the moment `N` rays have
                        terminated, which leaves a whole batch mid-path; their deposits are in the
                        tally but they were never counted in the denominator, so `E` came out high
                        by ~`n_batch/N`. Draining costs one extra batch of rays -- at production
                        ray counts a fraction of a percent of the run.
max_drain_batches       Guard on the drain: at most this many kernel launches, each of
                        `iterations_bunch` steps (default 10).
acc                   Element type of the accumulator arrays `E` and `dE` (default `Float64`).
verbose                 Print the run configuration and show a progress bar (default `true`).
stats                   When `true`, append a `NamedTuple` of run statistics to the return
                        value -- `(E, stats)` or `(E, dE, stats)`. It carries `n_rays`,
                        `n_stranded` (how many rays the drain guard cut short), `n_steps`,
                        `steps_per_ray`, `setup_time`, `transport_time`, `teardown_time`,
                        `wall_time`, `rays_per_s` and `steps_per_s`. `steps_per_ray` is the
                        key diagnostic when profiling: under delta tracking it is a property
                        of the physics and must not change when only the voxel count does.
block                   Edge length of the memory bricks the voxel arrays are stored in, one
                        of $(SUPPORTED_BLOCKS) (default `1`). `1` is plain column-major
                        storage. Larger values keep spatially neighbouring voxels close in
                        memory, which helps only if L2 misses are what limits the run --
                        see [`BrickLayout`](@ref) and `scripts/profile_grid.jl`.

# Example
    μ_a = randn(100, 10, 100)
    μ_s = randn(100, 10, 100)
    
    L₀ = BeamSampler(gaussian_beam_sampler((0., 0.)))

    light_transport_gpu(L₀, μ_a, μ_s, 0.9, (1., 0.1, 1.), ((-1., -0.1, 1.), (2., 0.2, 2.)), 100_000, 100)

"""
function light_transport_gpu(L₀::BeamSampler{T}, μ_a, μ_s, dμ_a, dμ_s, g, grid_size, simulation_bounds,
                             N, iterations_bunch;
                             n_threads=256, n_blocks=1024, max_loops=Inf,
                             term_threshold=1f-5, acc=Float64, verbose=true, stats=false,
                             block=1, drain=true, max_drain_batches=10) where T
    t_enter = time()
    @assert size(μ_a) == size(μ_s)
    # `K` is the number of derivative directions carried simultaneously. It becomes a
    # `Val` type parameter of the kernel so the per-direction loop unrolls and the tangent
    # state stays in registers as an `SVector{K}` rather than spilling to global memory.
    # A 3D `dμ_a` is the K == 1 case and is reported back as a 3D array.
    dμ_a, dμ_s, K, stacked = normalize_directions(μ_a, dμ_a, dμ_s)
    sensitivity = K > 0

    spatial_discretization = SA{T}[(grid_size ./ size(μ_a))...]
    simulation_bounds = Tuple(SA{T}[bounds...] for bounds in simulation_bounds)
    μ_max = T(maximum(μ_a .+ μ_s) * (sensitivity ? 1.2 : 1))
    if isa(g, Number)
        g = fill(T(g), size(μ_a))
    end
    @assert maximum(abs.(g)) <= 1 "Anisotropy must be in the interval [-1, 1]"
    n_threads, n_blocks, n_batch = cuda_params(N, n_threads, n_blocks)

    if verbose
        println("sensitivity=$(sensitivity) K=$(K)")
        println("dspace = $(spatial_discretization)")
        println("sim bounds = $(simulation_bounds)")
        println("μ_max=$(μ_max) max(μ_a)=$(maximum(μ_a)) max(μ_s)=$(maximum(μ_s)) min(g)=$(minimum(g))")
        println("threads=$n_threads blocks=$n_blocks batches=$n_batch")
    end

    # Every voxel-indexed array shares one layout, so a deposit lands in the same
    # brick the properties were just read from. The layout addresses exactly the
    # volume `pack_properties` produced, guard cells included -- it is told that
    # extent rather than deriving it, so it cannot disagree with the padding.
    props_host = pack_properties(T, μ_a, μ_s, g)
    layout = brick_layout(block, size(props_host)[2:4])
    props = CuArray{T}(brick_permute(props_host, layout))
    dμ_a, dμ_s = map(x -> sensitivity ? CuArray{T}(brick_permute(pad_stack(x), layout)) :
                                        CUDA.zeros(T, 1, 1),
                     (dμ_a, dμ_s))
    E = CUDA.zeros(acc, length(layout))
    dE = sensitivity ? CUDA.zeros(acc, K, length(layout)) : CUDA.zeros(acc, 1, 1)

    CUDA.@sync p_samples, d_samples, R_samples, dR_samples, dP_samples = initialize_samples(L₀, n_batch, K)
    terminate = CUDA.fill(false, n_batch)
    refill_buf = RefillBuffers{T}(n_batch)
    sample_pool = SamplePool{T}(n_batch)

    p = Progress(N, dt=0.5, showspeed=true, enabled=verbose)
    loops = 0
    n_finished_rays = 0
    eval_cuda_kernel!() = @cuda threads = n_threads blocks = n_blocks monte_carlo_kernel!(
        E, p_samples, d_samples, R_samples, terminate,
        props, μ_max,
        dE, dR_samples, dP_samples, dμ_a, dμ_s,
        simulation_bounds, spatial_discretization, layout,
        T(term_threshold),
        iterations_bunch, Val(K))

    verbose && @info "Start propagating $N rays"
    CUDA.synchronize()
    t_start = time()
    while n_finished_rays < N && loops < max_loops
        Δt = @timed begin
            eval_cuda_kernel!()
            # `@cuda` returns as soon as the launch is queued, so this host work
            # runs while the kernel is still executing.
            topup!(sample_pool, L₀)
            CUDA.@sync blocking=false begin
                refill_samples!(L₀, p_samples, d_samples, R_samples,
                                dR_samples, dP_samples, terminate, refill_buf, sample_pool)
            end 
        end
        # `@timed` carries the block's value through: `refill_samples!` returns the
        # number of slots that terminated, counted on the host while scanning the
        # bitmask, so this avoids a device reduction plus a sync each iteration.
        n_terminated = Δt.value
        n_finished_rays += n_terminated
        loops += n_threads * n_blocks * iterations_bunch
        update!(p, n_finished_rays,
                showvalues = () -> [(Symbol("Rays/s"), n_terminated / Δt.time),
                                    (Symbol("Energy"), "$(100 * sum(E) / n_finished_rays)%")])
        fill!(terminate, false)
    end
    CUDA.synchronize()

    n_stranded = 0
    if drain
        n_drained, n_stranded, residual, drain_batches =
            drain_in_flight!(eval_cuda_kernel!, terminate, R_samples, refill_buf.terminate;
                             max_batches = max_drain_batches,
                             progress_cb = (n_alive, _) -> update!(p, n_finished_rays,
                                 showvalues = () -> [(Symbol("Draining"), "$n_alive rays in flight")]))
        n_finished_rays += n_drained + n_stranded
        loops += drain_batches * n_threads * n_blocks * iterations_bunch
        verbose && finish!(p)
        if n_stranded > 0
            # `residual` and `sum(E)` are both unnormalised weight, so their ratio
            # bounds the fraction of the tally the truncation cost -- bounds rather
            # than equals, since some of that weight would have escaped the box
            # rather than being deposited.
            lost = 100 * residual / max(sum(E), eps(Float64))
            @warn("$n_stranded rays hit the drain guard and were cut short; they still " *
                  "carried up to $(round(lost, digits=4))% of the tallied energy. Raise " *
                  "`max_drain_batches` if that is too much.")
        end
    end
    transport_time = time() - t_start

    loops >= max_loops && @warn "Some rays were terminated as they exceeded the `max_loops` threshold"
    if verbose
        println("\n\n")
        @info("Total energy: $(100*sum(E)/n_finished_rays)%")
    end

    t_teardown_start = time()
    E_out = unbrick_unpad(E, layout, size(μ_a)) ./ n_finished_rays
    dE_out = sensitivity ?
        denormalize_directions(unbrick_unpad(dE, layout, size(μ_a)) ./ n_finished_rays,
                               stacked) : nothing
    run_stats = (n_rays = n_finished_rays,
                 n_stranded = n_stranded,
                 block = blocksize(layout),
                 overhang = overhang(layout),
                 n_steps = loops,
                 steps_per_ray = loops / max(n_finished_rays, 1),
                 setup_time = t_start - t_enter,
                 transport_time = transport_time,
                 teardown_time = time() - t_teardown_start,
                 wall_time = time() - t_enter,
                 rays_per_s = n_finished_rays / transport_time,
                 steps_per_s = loops / transport_time)

    if sensitivity
        return stats ? (E_out, dE_out, run_stats) : (E_out, dE_out)
    else
        return stats ? (E_out, run_stats) : E_out
    end
end


function drain_in_flight!(eval_kernel!, terminate, R_samples, host_terminate;
                          max_batches::Integer,
                          progress_cb = (n_alive, batches) -> nothing)
    n_batch = length(host_terminate)
    n_alive = n_batch
    batches = 0

    while n_alive > 0 && batches < max_batches
        eval_kernel!()
        # Counting terminate on the host gives the synchronisation the loop needs
        copyto!(host_terminate, terminate)
        n_alive = n_batch - count(host_terminate)
        batches += 1
        progress_cb(n_alive, batches)
    end

    residual = zero(eltype(R_samples))
    if n_alive > 0
        R_host = Array(R_samples)
        @inbounds for i in eachindex(host_terminate)
            host_terminate[i] || (residual += R_host[i])
        end
    end
    return n_batch - n_alive, n_alive, residual, batches
end
