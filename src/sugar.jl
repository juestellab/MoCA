"""
    sampler_from_weight_array(weights::Array{T, N}, distances::NTuple{N, T}) where {T, N}
Instantiate a `DirectSampler` object that samples position coordinates according to `weights`.
The argument `distances` is used to transform the index of an array-entry to a coordinate.
"""
function sampler_from_weight_array(weights::Array{T, N}, distances::NTuple{N, T}) where {T, N}
    w = Weights(weights[:])
    ρ_p(N) = begin
        idx = StatsBase.sample(1:length(weights), w, N)
        s1, s2 = size(weights)
        _p = Array{T, 2}(undef, 2, N)
        @. _p[1, :] = ((idx - 1) % s1 + 1 + $rand(T, N)) * distances[1] - s1/2 * distances[1]
        @. _p[2, :] = ((idx - 1) ÷ s1 + 1 + $rand(T, N)) * distances[2] - s2/2 * distances[2]
        return _p
    end
    return DirectSampler{T}(ρ_p)
end

"""
    gauss_profile_sampler(T, c, σ) 
Instantiates `DirectSampler` object that generates position samples centered at `c` with a standard-deviation `σ`.
"""
function gauss_profile_sampler(T, c, σ)
    c, σ = T.(c), T.(σ)
    @assert size(c, 1) == size(σ, 1)
    if typeof(σ) == Array{T, 2}
        @assert size(σ, 1) == size(σ, 2)
        return DirectSampler{T}(N -> repeat(c, 1, N) .+ σ * randn(T, size(σ, 1), N))
    elseif typeof(σ) == Vector{T}
        return DirectSampler{T}(N -> repeat(c, 1, N) .+ repeat(σ, 1, N) .* randn(T, size(σ, 1), N))
    end
end

"""
    focused_beam_direction_sampler(grid_size, dist_focus)
Direction of the rays are sampled according to a beam that is focused in the
center of the grid in the first two coorinates and `dist_focus` in the third.
"""
function focused_beam_direction_sampler(T, grid_size, dist_focus)
    c1 = T(grid_size[1] / 2)
    c2 = T(grid_size[2] / 2)

    isinf(dist_focus) && return p_list -> (res = zero(p_list); res[:, :, 3] .= -sign(dist_focus) * 1.0; return res)

    sampler(p_list) = begin
        N = size(p_list)[2]
        res = zeros(T, 3, N)
        x = p_list[1, :] .- c1
        y = p_list[2, :] .- c2
        r = @. sqrt(x^2 + y^2) + eps()
        # TODO @. not tested here
        θ = @. -atan(r / dist_focus)
        ϕ = @. sign(x) * acos(y / r)
        @. res[1, :] = sin(θ) * sin(ϕ)
        @. res[2, :] = sin(θ) * cos(ϕ)
        @. res[3, :] = cos(θ)
        return res
    end
    return DirectSampler{T}(sampler)
end

"""
    create_xy_beam_sampler(grid_size, initial_spot_size, focus_dist)
Creates a `BeamSampler` instance with a Gaussian beam profile in the xy plane
with standard-deviation `initial_spot_size`, focused at `focus_dist` in the
direction of movement (the z axis). For a diverging beam, `focus_dist` can be
set to a negative value.
"""
function create_xy_beam_sampler(T, grid_size, initial_spot_size, focus_dist)
    beam_profile = gauss_profile_sampler(T, [grid_size[1:2]...]./2, initial_spot_size)
    beam_direction = focused_beam_direction_sampler(T, grid_size, focus_dist)
    add_z_dim(x) = vcat(x, fill(0, (1, size(x, 2))))
    return BeamSampler(beam_profile, beam_direction, add_z_dim)
end


function normal_angle_sampler(T, σ_cone, tilt)::DirectSampler

    @inline function fun(p_membrane::Array{T, 2}) where {T}
        n_samples = size(p_membrane, 2)
        d′ = zeros(T, 3, n_samples)

        θ = T(σ_cone) .* randn(T, n_samples)  # into tissue ~N(0, σ_cone)
        ϕ = T(2π) .* rand(T, n_samples)  # x-y angle ~U[0, 2π)

        # draw ray in cone orthogonal to (x, y) plane
        d′[1, :] .= cos.(θ)  # angle about +z
        d′[2, :] .= sin.(θ) .* cos.(ϕ)  # angle about +x
        d′[3, :] .= sin.(θ) .* sin.(ϕ)  # angle about +y

        # angle ray in (z, y) space to emulate MSOT illumination
        d = similar(d′)
        d[1, :] .= cos(tilt) .* d′[1, :] - sin(tilt) .* d′[3, :]  # angle about +z
        d[2, :] .= d′[2, :]  # angle about +x
        d[3, :] .= sin(tilt) .* d′[1, :] + cos(tilt) .* d′[3, :]  # angle about +y

        d ./= sqrt.(sum(d.^2, dims=1))
        @assert all(isapprox.(sum(d.^2, dims=1), 1.0))
        # @assert all(d .>= 0)

        return d
    end

    sampler = DirectSampler{T}(fun)

    return sampler

end

# TODO: allow variable membrane. Not sure how to do that so it plays nice w/ Python.
# Can't cast python lambda to julia function. Probably should just pass a 2d mask and let 
# julia interp that. 
function membrane_shifter(z_membrane, lateral_shift, oop_shift)::Function

    @inline function fun(p_membrane::Array{T, 2}) where {T}
        n_samples = size(p_membrane, 2)
        p_volume = zeros(T, 3, n_samples)

        p_volume[1, :] .= T(z_membrane)
        p_volume[2, :] .= p_membrane[1, :] .+ T(lateral_shift)
        p_volume[3, :] .= p_membrane[2, :] .+ T(oop_shift)

        return p_volume
    end

    return fun

end


function embed_and_shift(_p, s)
    p = Array{Float32, 2}(undef, 3, size(_p, 2))
    p[1,:] .= s[1]
    p[2,:] .= _p[2,:] .+ s[2]
    p[3,:] .= _p[1,:] .+ s[3]
    return p
end


function angle_sampler(p, mean_angle, Δθ)
    Δd = zeros(3, size(p,2))
    θ = Δθ .* randn(size(p, 2))
    ϕ = 2π .* rand(size(p, 2))

    Δd[1, :] .= cos.(θ)
    Δd[2, :] .= sin.(θ) .* cos.(ϕ)
    Δd[3, :] .= sin.(θ) .* sin.(ϕ)

    d = similar(Δd)
    @. d[1, :] = sin(mean_angle) * Δd[2,:] + cos(mean_angle) * Δd[1, :]
    @. d[2, :] = cos(mean_angle) * Δd[2,:] - sin(mean_angle) * Δd[1, :]
    @. d[3, :] = Δd[3, :]
    return d
end

function initialize_beam_sampler(T, angle, beam_divergence, shift, illumination_profile_path)
    profile, res = h5open(illumination_profile_path) do file
        profile = T.(Array(file["profile"]))
        res = T(Array(file["px_size"])) * 100
        return profile, res
    end

    beam_sampler = BeamSampler(profile, (res, res), 
                               DirectSampler{T}(p -> angle_sampler(p, angle, beam_divergence)),
                               _p -> embed_and_shift(_p, shift))
    next(beam_sampler, 10)
    return beam_sampler
end
