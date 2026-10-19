################################################################################
# Ray supply: initial fill, host-side pool, and device refill
################################################################################
#
# How rays get onto the GPU and get replaced there once they terminate.
# The beam itself is described by `BeamSampler` in BeamSampling.jl; this file
# is only about staging its output for the kernel.

"""
    initialize_samples(beam_sampler, n_samples, K)
Draw positions, directions and energies for `n_samples` rays and initialize the differential
parameters. The tangent state `dR`, `dP` is `K x n_samples`: one column per ray, one row per
derivative direction. `K == 0` (no sensitivity) yields empty `0 x n_samples` arrays.
"""
function initialize_samples(beam_sampler, n_samples, K)
    p_samples, d_samples, R_samples = next(beam_sampler, n_samples)
    T = typeof(beam_sampler).parameters[1]
    return map(CuArray, (p_samples, d_samples, R_samples,
                         zeros(T, K, n_samples), zeros(T, K, n_samples)))
end

"""
    SamplePool{T}(capacity)

Host-side reservoir of pre-generated rays.

`next(beam, n)` is surprisingly expensive -- it allocates fresh `3 x n` position
and direction matrices plus a weight vector on every call, which for a 16k batch
costs ~300 us even for a trivial pencil beam and ~600 us for an isotropic one.
Because that work used to happen *after* the `terminate` bitmask was read back,
it was serialised against the GPU: measured at 65% of total runtime for the
semi-infinite phantom.

Sampling does not depend on ray state, so it can be done ahead of time. The pool
is topped up immediately after the kernel launch, while the GPU is still busy,
and drawn down when the refill happens.

**Nothing generated is thrown away.** The pool persists across iterations and is
consumed from the top, so whatever is left over after one refill is used by the
next; a top-up only ever generates the shortfall. The sole residue is whatever
remains in the pool when the run ends -- at most `capacity` rays, against a total
of `N`.
"""
mutable struct SamplePool{T}
    p::Matrix{T}
    d::Matrix{T}
    R::Vector{T}
    navail::Int
end
SamplePool{T}(capacity::Integer) where T = SamplePool{T}(
    Matrix{T}(undef, 3, capacity), Matrix{T}(undef, 3, capacity),
    Vector{T}(undef, capacity), 0)

"""
    capacity(pool::SamplePool)
How many rays the pool can hold. Derived from the backing arrays rather than
stored, so it cannot fall out of step with them.
"""
capacity(pool::SamplePool) = length(pool.R)

"""
    topup!(pool::SamplePool, beam_sampler)
Generate exactly the shortfall between `pool.navail` and `pool.capacity`.
Call this straight after the kernel launch so it overlaps GPU execution.
"""
function topup!(pool::SamplePool{T}, beam_sampler::BeamSampler{T}) where T
    k = capacity(pool) - pool.navail
    k <= 0 && return 0
    new_p, new_d, new_R = next(beam_sampler, k)
    slots = (pool.navail + 1):(pool.navail + k)
    copyto!(view(pool.p, :, slots), new_p)
    copyto!(view(pool.d, :, slots), new_d)
    copyto!(view(pool.R, slots), new_R)
    pool.navail += k
    return k
end

"""
    RefillBuffers{T}(n)

Scratch for [`refill_samples!`](@ref), allocated once per `light_transport_gpu`
call rather than once per outer iteration.

Only `terminate` ever travels device-to-host. The replacement rays are built on
the host, staged into the `new_*` device buffers, and scattered into place by a
kernel, so the ray state itself never leaves the GPU.
"""
struct RefillBuffers{T}
    terminate::Vector{Bool}     # host: the only device-to-host transfer
    idx::Vector{Int32}          # host: which slots terminated
    new_idx::CuVector{Int32}    # device staging for the above
    new_p::CuMatrix{T}
    new_d::CuMatrix{T}
    new_R::CuVector{T}
end
RefillBuffers{T}(n::Integer) where T = RefillBuffers{T}(
    Vector{Bool}(undef, n), Vector{Int32}(undef, n),
    CuVector{Int32}(undef, n), CuMatrix{T}(undef, 3, n), CuMatrix{T}(undef, 3, n),
    CuVector{T}(undef, n))

"""
    scatter_refill_kernel!(p, d, R, dR, dP, new_p, new_d, new_R, idx, n_new)
Write the `n_new` freshly sampled rays into the slots named by `idx`, resetting
the differential state of those slots to zero.
"""
function scatter_refill_kernel!(p, d, R, dR, dP, new_p, new_d, new_R, idx, n_new)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= n_new
        @inbounds begin
            j = idx[i]
            p[1, j] = new_p[1, i]; p[2, j] = new_p[2, i]; p[3, j] = new_p[3, i]
            d[1, j] = new_d[1, i]; d[2, j] = new_d[2, i]; d[3, j] = new_d[3, i]
            R[j] = new_R[i]
            for k = 1:size(dR, 1)
                dR[k, j] = zero(eltype(dR))
                dP[k, j] = zero(eltype(dP))
            end
        end
    end
    return nothing
end

"""
    refill_samples!(beam_sampler::BeamSampler{T}, p, d, R, dR, dP, terminate, buf)
If an entry in `terminate` is true, replace the corresponding entries in
`p`, `d` with a new sample and reset `R`, `dR`, `dP` to zero.

Mutates the device arrays in place and returns the number of rays replaced.

Only the `terminate` bitmask is read back from the device; the new rays are
taken from `pool` (see [`SamplePool`](@ref)), uploaded as a compact `3 x n_new`
block, and scattered by [`scatter_refill_kernel!`](@ref). The ray state arrays
are never downloaded, which is what keeps the per-iteration host cost from
scaling with `n_batch`.
"""
function refill_samples!(beam_sampler::BeamSampler{T}, p, d, R, dR, dP, terminate,
                         buf::RefillBuffers{T}, pool::SamplePool{T}) where T
    copyto!(buf.terminate, terminate)

    n_new_samples = 0
    @inbounds for i in eachindex(buf.terminate)
        if buf.terminate[i]
            n_new_samples += 1
            buf.idx[n_new_samples] = i
        end
    end
    n_new_samples == 0 && return 0

    # The pool is topped up to capacity >= n_batch before the sync, and at most
    # n_batch slots can terminate, so this should never fire; it is a guard, not
    # a code path.
    pool.navail < n_new_samples && topup!(pool, beam_sampler)

    # Consume from the top so the untouched remainder stays in place and is used
    # by the next refill -- no sample is generated and then discarded.
    off = pool.navail - n_new_samples
    copyto!(buf.new_p, 1, pool.p, 3 * off + 1, 3 * n_new_samples)
    copyto!(buf.new_d, 1, pool.d, 3 * off + 1, 3 * n_new_samples)
    copyto!(buf.new_R, 1, pool.R, off + 1, n_new_samples)
    copyto!(buf.new_idx, 1, buf.idx, 1, n_new_samples)
    pool.navail = off

    threads = 256
    blocks = cld(n_new_samples, threads)
    @cuda threads = threads blocks = blocks scatter_refill_kernel!(
        p, d, R, dR, dP, buf.new_p, buf.new_d, buf.new_R, buf.new_idx, n_new_samples)
    return n_new_samples
end
